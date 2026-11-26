# Qwen3.8-27B

`models/qwen3_8_27b/` implements the W8A8 Gated DeltaNet block, including
checkpoint-compatible projections and separate convolution and recurrent
state for each packed request.

Qwen3.8-27B interleaves three Gated DeltaNet layers with one gated-attention
layer, `LLLA` repeated sixteen times over 64 layers, so 48 of its layers run the
operators in this directory.

## Model configuration

[config.py](../../../models/qwen3_8_27b/config.py) carries the values from the
published `config.json` under their HuggingFace names, so they can be checked
against it without a translation table.

| Property | Value |
| --- | --- |
| Hidden size | 5120 |
| Layers | 64 — 48 linear attention, 16 full attention (`full_attention_interval = 4`) |
| Linear-attention heads | 48 value, 16 QK (`linear_num_value_heads`, `linear_num_key_heads`) |
| Linear-attention head dim | 128 for both key and value |
| Short convolution | depthwise, kernel 4, over all 10240 q/k/v channels |
| Chunk length | 128 — our tiling choice, not a model constant |
| Platform | Ascend A2/A3 (`-p a2a3`); A5 serving is not validated |
| Quantization | W8A8 QKV and Z projections with per-channel INT8 weights and per-token INT8 activations. The A/B projections, gated RMSNorm output, and output projection use BF16. |

Grouped-query attention is the shape that matters here: value head `h` reads key
head `h // 3`. q and k carry 16 heads; v, beta, the gate, the state and the
output are per value head.

## The operators

The layer's core is the chunked delta rule, split the way the reference
PTO-ISA implementation splits it. Each file is a standalone entry point. The
first six are the delta rule; the rest are the block around it, in the order
the block runs them.

| Operator | What it computes |
| --- | --- |
| [chunk_cumsum.py](../../../models/qwen3_8_27b/chunk_cumsum.py) | chunk-local prefix sum of the gate logits |
| [scaled_dot_kkt.py](../../../models/qwen3_8_27b/scaled_dot_kkt.py) | the gated key-key matrix `A`, strictly lower triangular |
| [solve_tril.py](../../../models/qwen3_8_27b/solve_tril.py) | `(I + A)^-1` per chunk, with FP32 diagonal inversion and block substitution |
| [wy_fast.py](../../../models/qwen3_8_27b/wy_fast.py) | the WY representation, `W` and `U` |
| [chunk_h.py](../../../models/qwen3_8_27b/chunk_h.py) | the inter-chunk state recurrence and `V_new` |
| [chunk_o_pipe.py](../../../models/qwen3_8_27b/chunk_o_pipe.py) | the chunk output, inter-chunk plus intra-chunk, as an explicit cube/vector pipeline |
| [quant_x.py](../../../models/qwen3_8_27b/quant_x.py) | per-token int8 quantisation of the hidden states, shared by the projections that read them |
| [in_proj_qkv.py](../../../models/qwen3_8_27b/in_proj_qkv.py) | hidden states to q, k and v, A8W8 |
| [in_proj_z.py](../../../models/qwen3_8_27b/in_proj_z.py) | hidden states to the gated norm's gate, A8W8 |
| [in_proj_ab.py](../../../models/qwen3_8_27b/in_proj_ab.py) | hidden states to `a` and `b`, the two fused into one bf16 matmul |
| [short_conv.py](../../../models/qwen3_8_27b/short_conv.py) | the depthwise causal convolution over the q/k/v channels, with silu |
| [qk_norm_gate.py](../../../models/qwen3_8_27b/qk_norm_gate.py) | the q/k L2-norm and scaling, and `beta` and the decay gate from the a/b projections |
| [gated_rmsnorm.py](../../../models/qwen3_8_27b/gated_rmsnorm.py) | the gated RMSNorm after the delta rule, in bf16 for the block and with a per-token INT8 epilogue for an A8W8 reader |
| [out_proj.py](../../../models/qwen3_8_27b/out_proj.py) | the normed output back to the residual stream, BF16 both sides |
| [gdn_layer.py](../../../models/qwen3_8_27b/gdn_layer.py) | the delta rule's six stages as one program |
| [gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py) | all fourteen as one program, hidden states in and out |

The two A8W8 projections are the same arithmetic at two shapes and share
one implementation, [a8w8_linear.py](../../../models/qwen3_8_27b/a8w8_linear.py);
each instance file supplies its shape, its tiles and where its weight and
references come from.

`solve_tril` is the delta-rule triangular inversion of
[arXiv:2605.21325](https://arxiv.org/abs/2605.21325).

FP32 inversion and block substitution preserve precision for correlated keys,
including repeated-token prompts. The kernel publishes the inverse in FP16.

## The layer

[gdn_layer.py](../../../models/qwen3_8_27b/gdn_layer.py) composes all six as one
forward. `A`, `A_inv`, `W`, `U`, the per-chunk state snapshots and `V_new` are
internal. The caller supplies q, k, v, the gate, beta, and each request's
recurrent state row. The layer returns its output and updates the recurrent
state in place. The layer output and state are scored against the reference.

The standalone and composed paths use the same operators. Both publish the
triangular inverse as FP16, which is the dtype consumed by `wy_fast`;
intermediate inversion arithmetic remains FP32.

`gdn_layer.py` is the delta rule alone. The surrounding block is
[gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py), below.

## The block

[gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py) runs all fourteen
operators as one program: bf16 hidden states in, bf16 hidden states out, with
every intermediate allocated inside and never reaching the caller. The delta
rule inlines into it as one callee in turn, so the six stages inline into
`gdn_layer` and `gdn_layer` inlines here.

Three interfaces are shaped by the composition rather than by any one operator,
and each is answered where it costs nothing:

- The convolution reads the K-1 tokens before each of its windows, so
  `in_proj_qkv` takes a `row_off` and writes straight into the padded buffer
  the convolution wants; the rows above it are zeroed by a ten-task fill.
  Nothing copies `[T, C]` into `[T + K - 1, C]`.
- The convolution writes q, k and v as three tensors rather than one, so each
  consumer reads the rows of a contiguous buffer -- and v, which goes straight
  to the delta rule, is written in fp16 at the store instead of by a later
  cast pass.
- The gated norm declares `z` in the `[T, H*D]` layout `in_proj_z` writes.

No slicing is needed at any call site: every delta-rule kernel already
reshapes its rank-3 inputs to `[T, heads*D]` as its first statement, so what
the consumers need is each tensor contiguous in its own buffer, not a
particular rank.

Composing the fourteen into one program is what lets the runtime overlap them,
and the gain is concentrated at the head, where `in_proj_ab`, the halo fill,
the gate half of `qk_norm_gate` and `chunk_cumsum` all run underneath
`quant_x`.

**Accuracy is scored as ratios against the quantization floor, not as an
element-wise tolerance**, because the block quantizes the hidden states per
token: a difference far below the int8 step flips values that sit near a
rounding boundary, and every projection downstream then reads a different
integer -- so an element-wise comparison taken after a quantiser measures
which side of a boundary a value fell on, not whether the kernel is right.
The gate is instead total relative error within 1.05x of what W8A8 alone
costs and worst token row within 1.25x, plus the same two distances to the
fake-quant chain itself. The worst row is the sharper of the two: one dead
row moves it by an order of magnitude where the whole-tensor norm barely
registers the same row.

Packed GDN uses separate convolution halos and recurrent state rows for each
request. The caller provides the row boundaries and cache owners.

## The short convolution

[short_conv.py](../../../models/qwen3_8_27b/short_conv.py) is the depthwise
causal convolution that follows the q/k/v projection: four taps over the token
axis per channel, fp32 accumulate, silu fused in the epilogue, no bias.

Its input is `[T + K - 1, C]`, carrying the K-1 tokens before the sequence in
its first rows. That is what keeps every tap offset non-negative, so no block
needs a head-padding special case; for prefill those rows are zero, and they
are where a decode cache would sit.

On a2a3 at T = 8192 over all 10240 channels, 50 rounds: **648 us**, 517 GB/s of
GM traffic against a measured copy roof of 1236 GB/s. CANN's own depthwise
`conv1d` plus silu takes 1532 us channels-first and 2597 us once the transpose
our token-major layout would need is charged, so 2.4x and 4.0x. Accuracy
against the float64 reference is max abs 2.6e-02 on the real layer, inside the
6e-02 the hand-written PTO-ISA kernel's own tests use for bf16.

## The qk-norm and gate

[qk_norm_gate.py](../../../models/qwen3_8_27b/qk_norm_gate.py) produces the
four things the delta rule reads: q and k L2-normalised over the head dim with
the epsilon inside the square root and q scaled by `D^-0.5`, `beta` as a
sigmoid of one small projection, and the decay `g` from the other.

`beta` comes out head-major, `[H, T]`, because that is what `wy_fast` and
`scaled_dot_kkt` declare; the kernel transposes on chip rather than leaving it
to the host. `softplus` is written as `log(1 + exp(x))` with the exponent
clamped, since the naive form overflows fp32 and this model's `dt_bias`
already reaches 19.25.

On a2a3 at T = 8192, 50 rounds: **122 us**, 1135 GB/s against a measured copy
roof of 1236 GB/s, so this kernel is bandwidth-bound. Accuracy against the
float64 reference is 1.6e-03 relative Frobenius on q and k, and better on
`beta` and `g`.

## The gated RMSNorm

[gated_rmsnorm.py](../../../models/qwen3_8_27b/gated_rmsnorm.py) is the norm
that follows the delta rule, `y = o * rsqrt(mean(o^2) + eps) * norm_w * silu(z)`
per (token, head) over the head dim, with the cast sequence the model's own
`Qwen3_5RMSNormGated` uses: fp32 variance and rsqrt, narrow to the model dtype,
apply the weight there, apply the gate in fp32.

The block takes the form that ends in the module's own cast, `out_proj` being
bf16 both sides. The file also carries an epilogue that quantises the result
per token over the whole 6144-wide row -- what an A8W8 reader of this output
needs, which is how the block read it before the BF16 `out_proj` -- and
`--bf16-out` selects between the two.

## The projections

Two of the four projections around the delta rule are W8A8: `in_proj_qkv` and
`in_proj_z`, with int8 weights at one scale per output channel, int8
activations at one scale per token, INT32 accumulation over a pipelined K loop,
dequantised by the two scales and written as bf16. The other two are bf16 --
`in_proj_ab` because int8 there buys nothing (below), and `out_proj` because
the deployed checkpoint publishes that weight unquantized.

[quant_x.py](../../../models/qwen3_8_27b/quant_x.py) does the activation side
once for everything that reads the hidden states, and it is the block's only
activation quantisation. The gated norm has the same arithmetic available in
its own epilogue, for an A8W8 reader of its output; the block does not use it.

[a8w8_linear.py](../../../models/qwen3_8_27b/a8w8_linear.py) is the matmul.
Its one structural choice is that a block walks an `m_group` by `n_group`
**patch** of output tiles rather than a single tile. One output tile is capped
near 128x128 by the cube-to-vector crossing -- the cross-core ring is sized by
the INT32 accumulator and charged to the same vector buffer that holds the
working copy -- and at one tile per block `in_proj_qkv` is 5120 tasks, more
than the runtime's ring heap carries. A patch fixes both: it cuts the task
count and it is what a bigger tile would have bought, since a block reads
`(m_group * m_tile + n_group * n_tile) * K` bytes for
`m_group * n_group` tiles of output. Square patches win; the shape alone is
worth 1.4x at identical tiles.

The cap is on the tile that *crosses*, though, not on what the cube may work
on, so the inner loop takes two n-tiles against one activation tile and drains
them separately -- the cube sees a 128x256 region from a single L1 operand
while each crossing stays 128x128. That is worth another 6 to 8%.

[in_proj_ab.py](../../../models/qwen3_8_27b/in_proj_ab.py) stays bf16 and fuses
`in_proj_a` and `in_proj_b` into one `[96, 5120]` weight: N = 48 alone is legal
under the multiple-of-16 tile rule but a poor cube tile, and two passes over
the hidden states would cost twice what one does.

On a2a3 at T = 8192, 50 rounds, against `torch_npu` 2.10.0 on the same card:

| kernel | PyPTO us | CANN operator | CANN us |
| --- | ---: | --- | ---: |
| `quant_x` | 191 | `npu_dynamic_quant` | 105 |
| `in_proj_qkv` | 1813 | `npu_quant_matmul` | 1660 |
| `in_proj_z` | 1014 | `npu_quant_matmul` | 993 |
| `in_proj_ab` | 115 | `F.linear` bf16 | 47 |

The two GEMMs run at 474 and 508 INT8 TOPS, about 1.7x their own bf16
`F.linear` at the same shapes and 0.92 to 0.98x of CANN's dedicated W8A8
operator. A
control that drops the dequantisation entirely measures within the noise of the
full kernel, so that gap is the cube schedule under a 128x128 output tile, not
the epilogue. Accuracy: each GEMM sits 1.7e-03 from the float64 W8A8 chain it
implements -- bf16 output rounding and nothing else -- and the quantisation is
the whole of the rest.

## Validating and benchmarking

Each operator file provides a golden and validation entry. The delta-rule
entries share the Torch stage operations in
[delta_rule_reference.py](../../../models/qwen3_8_27b/delta_rule_reference.py)
and check device results against their goldens.
[reference.py](../../../models/qwen3_8_27b/reference.py) holds what they share: the
drawn inputs, the two layout conversions and the acceptance criterion.
A default invocation runs each entry's own bounded correctness cases, and `--seq-len`
runs one shape instead. `PYPTO_BENCH=1` turns any of them into a benchmark, and the
harness prints the median `effective_us`.

Around those six, `reference.py` carries the whole block: the four projections, the
depthwise convolution, the qk-norm and gate, the gated RMSNorm and `out_proj`, read
line by line from `Qwen3_5GatedDeltaNet` in `transformers`. That block chain is what
every operator and the composed block are scored against, so it
is itself checked against the model's own module:
[test_block_reference.py](../../../models/qwen3_8_27b/test_block_reference.py)
runs `Qwen3_5GatedDeltaNet` on the same hidden states and weights and compares
the two at every projection, after the delta rule, after the norm and at the
output. With the module in float64 the projections agree exactly and everything
after the gate to about 3e-7 -- the module computes its gate, delta rule and norm
statistics in fp32 by its own code, which is the floor. In the model's own bf16
the block output sits 3e-3 to 7e-3 from the float64 chain, which is the
yardstick for what a device-side block can be held to.

The block has a float64 reference and a fake-quant reference. The fake-quant
reference uses per-channel INT8 QKV and Z weights and per-token INT8 hidden
states. The gated norm and output projection use BF16. Reference validation
measures device error against both the fake-quant output and the float64
reference. `test_block_reference.py --quant` also measures the effect of
quantizing the normed activations separately.

Two weight sets serve that check and every later one: random weights drawn the
way the module's own `__init__` draws them, and one real layer.
[weights.py](../../../models/qwen3_8_27b/weights.py) fetches a layer's nine
`linear_attn` tensors from the published checkpoint by range request, so no
checkpoint download is needed. A trained layer's `A_log` and `dt_bias` reach a
decay range the random init does not, and that is the range where a wrong
scale, a wrong eps or a saturating gate would show.

```bash
python models/qwen3_8_27b/gdn_block.py -p a2a3 -d "$TASK_DEVICE"
python models/qwen3_8_27b/gdn_layer.py -p a2a3 -d "$TASK_DEVICE"
python models/qwen3_8_27b/chunk_h.py -p a2a3 --sim
python models/qwen3_8_27b/solve_tril.py -p a2a3 --sim --seq-len 1024
PYPTO_BENCH=1 python models/qwen3_8_27b/gdn_layer.py -p a2a3 -d "$TASK_DEVICE" --seq-len 8192
```

The standalone block validates a fresh sequence and a resumed one with two
requests. `--packed` selects packed validation, whose total is 2048-token
aligned and whose requests are 256-token aligned. `--weights` takes the
original floating-point layer, which is what the quantization floor is
measured against.

The block reference check needs no NPU, only `transformers` (5.17.0), and skips with a message if it is absent. `weights.py` needs `--fetch` the first time, since it pulls from the checkpoint:

```bash
python models/qwen3_8_27b/weights.py --layer 0 --fetch
python models/qwen3_8_27b/weights.py --quantize
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192 --weights build_output/qwen3_8_27b/linear_attn_layer0.pt
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192 --weights build_output/qwen3_8_27b/linear_attn_layer0.pt --quant
python models/qwen3_8_27b/test_block_reference.py --module-dtype float64,bfloat16
```

The example commands invoke validation scripts directly. Device validation
compiles and runs the kernel. The CPU block reference defaults to random
weights and accepts an extracted layer for real-weight validation.
