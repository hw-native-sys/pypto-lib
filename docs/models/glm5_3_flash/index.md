# GLM-5.3-Flash

`models/glm5_3_flash/` is the implementation staging area for
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`glm5_next`,
`Glm5NextForConditionalGeneration`) on **Ascend a2a3**. This first milestone
establishes the configuration, the layer schedule, the metadata lowering, the
shared Torch goldens, the W8A8 references, and one `@pl.jit.inline` ABI per
operator. Kernel bodies are the parallel work that follows.

## Why W8A8, and why a2a3 forces it

The released checkpoint is FP8 blockwise (`e4m3`, 128x128, dynamic activations).
The a2a3 cube cannot execute it: `Intrinsic_mmad` in
CANN's `Ascend910_9392.ini` SoC config offers only `s32s8s8` / `u32u8u8` / `u8s8` and the
fp16/fp32 forms, with no fp8 or MX entry. INT8 is therefore the only quantized
path on this platform, and the deployment weights are the msmodelslim W8A8
conversion that vLLM Ascend serves on a 16-card A3 node. The int8 cube fractal
is MKN `16, 32, 16`, so every quantized K tile must be a multiple of 32.

## Checkpoint shape

| Property | Value |
| --- | ---: |
| Hidden size | 4,096 |
| Backbone layers | 45 (+1 MTP layer) |
| Linear-attention (KDA) layers | 34 |
| Sparse-attention (NoPE MLA + kpool indexer) layers | 11 (+ the MTP layer) |
| Attention heads | 64 |
| MLA `q_lora_rank` / `kv_lora_rank` | 1,536 / 512 |
| MLA `qk_head_dim` / `v_head_dim` / `qk_rope_head_dim` | 256 / 256 / **0** |
| Indexer heads / head dim / `index_topk` / `index_kpool` | 32 / 128 / 2,048 / 4 |
| KDA heads / head dim / conv kernel | 64 / 128 / 4 |
| Routed experts / active experts / shared | 288 / 8 / 1 |
| Dense layers / dense intermediate | 3 / 12,288 |
| MoE intermediate | 2,048 |
| Hyper-connection width / Sinkhorn iterations | 4 / 20 |
| Vocabulary | 154,880 |
| Maximum position | 1,048,576 |

The attention schedule is three KDA layers then one sparse layer, repeating:
layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 and 43 are sparse, every other
backbone layer is KDA. The first three layers (all KDA) use a dense MLP;
the remaining 42 and the MTP layer are MoE. Checkpoint layer 45 is the MTP
layer: an MLA layer with its own indexer and a sparse MoE, reached through
`enorm` / `hnorm` / `eh_proj`, and it carries **no** `hc_*` weights, so it uses a
plain residual rather than the four-stream mHC.

Summing every parameter group from the checkpoint index gives 321.2 G, which
matches the published 320 B total, so the reconstruction above is complete.

## What differs from DeepSeek-V4, which is the nearest port in this repo

`Glm5NextTextHyperConnection` subclasses `DeepseekV4HyperConnection` unchanged, so
the mHC coefficient algebra is identical to `models/deepseek_v4_1_flash/mhc.py`.
Three things are not:

- **The hyper-connection stream is BF16.** It is seeded by replicating the BF16
  embedding (`inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)`) and every layer
  casts the mixes into that dtype. DeepSeek-V4 carries the stream in FP32, so the
  tile dtypes differ throughout and a kernel cannot be lifted across unchanged.
  At width 4 this is 32 KB of live residual per token instead of 64 KB.
- **The final collapse is an unweighted mean**, not a learned head applying the
  last layer's delayed pre-mix, so `mhc_head` takes no mix input.
- **There is no RoPE anywhere.** `qk_rope_head_dim` is 0 and the config validator
  rejects any other value, so the rope/nope split disappears from the main
  attention — and the indexer has none either, despite `indexer_rope_interleave`
  being set in `config.json`. That field is inherited from the GLM-MoE-DSA base and
  is dead here: `Glm5NextTextConfig` sets `rope_parameters = AttributeError()`,
  `Glm5NextTextModel.forward` passes `position_embeddings=None`, and
  `Glm5NextTextIndexer.forward` never touches a cos/sin table. So this port needs
  none of the `rope_tables.py` / interleaved-rope machinery that both
  `deepseek_v4_pro` and `deepseek_v4_flash_mtp` carry.

Two more differences bite at the kernel level: the router scores with `sigmoid`
where DeepSeek-V4.1 uses `sqrt(softplus(.))`, and 288 experts do not fit the
`SCORE_PAD = 256` sort window that `models/deepseek_v4_flash_mtp/gate.py` uses —
the pad has to grow to 512 and the `pl.sort32` + `pl.mrgsort` merge tree gains a
level.

## Deployment target

One Atlas 800 A3 node, 16 dies, TP16 with expert parallelism over the same 16
ranks, mirroring vLLM Ascend's recipe for this checkpoint
(`--tensor-parallel-size 16 --enable-expert-parallel --quantization ascend`,
MTP with 3 speculative tokens, `FULL_DECODE_ONLY` graph capture). Smaller TP/EP
shapes stay available for bring-up through `--tp` and `--ep`.

Per rank that gives 4 MLA heads, 4 KDA heads, 18 routed experts and 9,680
vocabulary rows. The **indexer is replicated, not head-sharded**: its score sums
over all 32 heads before the top-k, so sharding would force a cross-rank reduction
of partial scores on every sparse layer, and the kernel is small enough that 16-way
redundant compute is the cheaper trade. Weights come to about 21.6 GB per rank, of
which 19.5 GB is the routed experts; the remaining budget on a 64 GiB die
(~61.3 GiB visible) goes to the hybrid cache, which is the binding constraint and
is sized by the cache-manager work item.

## What is quantized

Activations are quantized **dynamically per token** the way the a2a3 W8A8 sibling
does it: each row's amax, floored by `INT8_AMAX_EPS = 1e-4`, is rescaled to
`INT8_SCALE_MAX = 127`, rounded through int32 and fp16 to match device rounding.
The router kernel produces that INT8 view once and both the shared expert and the
EP dispatch payload reuse it.

The boundary below is read from the deployment checkpoint
(`Eco-Tech/GLM-5.3-Flash-w8a8` on modelers.cn), not inferred: its
`quant_model_description.json` labels every tensor except `rot.weight`, and the
shard headers confirm the dtypes. The split is sharper than the released FP8 checkpoint's.

| Tensor | Format |
| --- | --- |
| `mlp.experts.N.{gate,up,down}_proj`, `mlp.shared_experts.*`, and the three dense layers' `mlp.{gate,up,down}_proj` — **the FFN and nothing else** | **INT8**, with an FP32 `[out, 1]` per-output-channel `weight_scale` |
| **All of attention**: every MLA projection (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), every KDA projection and conv, every indexer weight, all mHC tensors, `mlp.gate.weight`, `e_score_correction_bias`, every norm, `embed_tokens`, `lm_head`, and the whole vision tower | BF16 |
| MLA 512-latent cache | BF16 |
| Indexer state cache | FP32 today, 256 wide per token; the sibling port quantizes its indexer cache to INT8 with a per-row FP32 scale, which would save about 2.3 KB per token here |
| Hyper-connection stream | BF16 — see above, this is where GLM departs from DeepSeek-V4's FP32 stream |
| Logits | FP32 |

Of 113,352 labelled tensors, 111,870 are `W8A8_DYNAMIC` and 1,482 are `FLOAT`. Each
quantized weight also carries a `weight_offset`, but it is **all zero** — measured
across tensors of 12,288, 2,048 and 4,096 channels — so the weight quantization is
symmetric and the loader reads and discards the offset. There is no static
activation path and no SmoothQuant vector.

The consequence for the work list is that W8A8 is confined to the MoE and dense-MLP
streams. Attention carries no quantization at all, which removes the scale plumbing
from `mla_prolog`, `mla_epilog` and the indexer projections.

### The checkpoint also ships two matrices that are not weights

`rot.weight`, BF16 `[4096, 4096]`, is the only tensor in the weight index with no
entry in the quantization description; it is announced solely by `is_rot_used:
true`. It belongs to the **MTP layer**, not the backbone — vLLM Ascend's
`AscendDeepSeekMTP` applies it to the previous hidden state before `hnorm`. GLM
routes to `Glm5NextMTP` instead, which has no `rot` and drops the tensor silently,
so whether it is required is an open question the MTP owner inherits. It is not a
rotation, despite the name: it is symmetric with `R[i,j] = g(i xor j)`, a
per-channel diagonal scaling in the Hadamard basis.

`optional/quarot.safetensors` holds `global_rotation`, F32 `[4096, 4096]`, an exact
scaled Hadamard (every entry +/-1/64, rows orthonormal) — a textbook QuaRot matrix.
It is unreachable for this checkpoint: vLLM Ascend looks it up under
`quant_description["optional"]["quarot"]`, and this description's `optional` is
empty. Where that matrix is used elsewhere in vLLM Ascend it is folded into weights
offline, never applied to activations. **No backbone kernel applies a rotation.**

The one rotation that does matter is 128 wide and lives in the indexer: the upstream
kernel rotates each query head by a Hadamard-128 before quantizing it. See the
indexer files.

## Prior art, and what it is actually worth

None of the references below is drop-in code for this repo, but two of them are
worked designs for the hardest kernels on our exact platform.

| Reference | Where | What it gives us |
| --- | --- | --- |
| Chunked KDA in pypto, tuned on Ascend910B3 | cann-recipes-infer, `integration/vllm/ling-3.0-flash/npu_patch/.../ops/pypto/kda/chunk_kda_impl.py` | A complete chunked delta-rule forward at `T=4K, H=4, K=V=128` — our per-rank TP16 shape — with a design write-up covering the M1-M4 stages, the 8x8-leaf block inverse, and the two bottlenecks whose fix was worth 3.8x |
| Recurrent KDA in pypto | its `fused_recurrent_kda_impl.py` sibling | The decode-step counterpart |
| Lightning indexer, MLA prolog, hc_pre, sparse attention in pypto | cann-recipes-infer, `ops/pypto_python/impl/` | Algorithm and tiling references for the indexer and MLA items |
| vLLM Ascend `glm5next` | `vllm_ascend/models/glm5next/` | The only NPU port of this architecture: module decomposition, cache specs, W8A8 wiring, MTP. **It does not implement the kpool indexer** — `sparse_attn_indexer_kpool.py` raises and says the upstream is CUDA-only |
| CANN recipes GLM-5 / GLM-5.2 | cann-recipes-infer, `models/glm_5{,_2}/` | Ascend-native MLA + DSA + MoE with a 16-rank W8A8 A3 config. No KDA and no mHC |
| GLM-4.5 pypto operators | the open-source pypto mirror, `models/glm_v4_5/` | MoE gate, expert selection and the shared-expert W8A8 patterns |

The two pypto references above target a **different pypto frontend**
(`pypto.frontend.jit` / `pypto.Tensor` / `pypto_impl` / auto-tiling). The pypto
this repo pins exports only `jit`, `language`, `ir`, `runtime` and friends, so
those files are design references, not code to copy: the port is from implicit
tiling to the explicit `pl` tile DSL.

## Platform envelope

| a2a3 (`Ascend910_9392`) | Value | a5 for contrast |
| --- | ---: | ---: |
| AIC / AIV per die | 24 / 48 | 36 / 72 |
| `PLATFORM_MAX_BLOCKDIM` / max cores | 24 / 72 | 36 / 108 |
| UB (safe) | 192 KB (~184 KB) | 248 KB |
| L1 / L0A / L0B / L0C | 512 / 64 / 64 / 128 KB | — |
| HBM per die | 64 GiB (~61.3 GiB visible) | — |
| fp8 / MX `mmad` | **none** | supported |
| 4-bit in-core dtypes | **not supported** | FP4 supported |
| `Acc -> Vec` and `Vec -> Mat` memory edges | **absent** | present |
| AIC/AIV cross-core pipe | through GM | on-chip |
| GM access granularity | 512 B | 128 B |
| bf16 atomic add into GM | supported | not supported |

Some of these change how a kernel is written rather than how fast it runs. The
missing `Acc -> Vec` edge forces a GM round trip where a5 would fuse. INT8 can only
reach GM through a Vec tile or the quantized fix-pipe path. And not in the table but
worth knowing before the first MIX kernel: a2a3 requires dual-AIV dispatch even for
a no-split MIX kernel, or the AIC cross-core handshake deadlocks.

### What stream B's `a2a3sim` bring-up surfaced

Each of these was a compile or simulator failure, not a guess:

- **The simulator needs a real GCC 15.** A `g++-15` shim pointing at GCC 13 builds
  simulator kernels that get every BF16 matmul wrong while FP32 still passes, so the
  failure looks like a kernel bug. `deepseek_v4_flash_mtp/decode_compressor_ratio4.py`
  is a quick canary: it passes in CI and fails under such a shim.
- **Do not mix the Tensor and Tile levels.** `pl.slice`, bracket slices, `pl.full`,
  `pl.matmul_acc` and slice assignment are Tensor level; `pl.load`, `pl.tile.full`,
  `pl.create_tile` and `pl.store` are Tile level. A Tensor-level result is written with
  `dest = pl.assemble(dest, pl.set_validshape(src, rows, cols), offset)`, a Tile-level
  one with `pl.store`; `pl.write` works on a Tensor-level on-core buffer where
  `pl.tile.write` does not.
- **`tmp_tile` is Tile-only.** `pl.row_sum` / `pl.row_max` on a Tile must take one;
  on a Tensor they must not. High-precision `rsqrt` on a Tile is
  `pl.tile.rsqrt(x, tmp)`, not `high_precision=True`.
- **FP32 reductions need a 32-byte column**, and a row-major tile needs a 32-byte row,
  so an `[H, 1]` column wants `H % 8 == 0` and is best built as `[1, H]` then
  reshaped.
- **A matmul is only split into L0 tiles when M reaches 16.** Below that the
  128 x 512 KV operand of the sparse attention stays whole and overflows the 64 KB
  right buffer; the TP16 attention pads its 4 heads to 16 zero-filled rows for this.
- **No mutable scalar flags across branches.** A Python int set in one `if` and read in
  another becomes a phi the backend cannot materialise; write each branch's work in
  place, as the donors do. Loop-carried tile state goes through `pl.yield_` on every
  path.
- **In a mixed cube/vector scope, no branch may decide whether loop-carried state is
  updated.** When the carried tiles flow through an `if` / `else`, the partitioner
  materialises their pre-loop `pl.full` initialisation in the cube half as well, and
  `ccec` rejects `vector_dup` / `set_vector_mask` for `dav-c220-cube`. `a2a3sim`
  compiles both halves with `g++` and never sees it, so this reaches CI as an a2a3-only
  failure (run 35555578466). Let the branch pick the block's inputs and keep the merge
  unconditional, or split the scope by hand with `pl.split_aic` / `pl.split_aiv` as
  `deepseek_v4_flash_dspark/prefill_sparse_attn.py` does. The check needs no card:
  compile with `-p a2a3 --compile-only` and assert that the generated
  `kernels/aic/*.cpp` carries no vector op between `#if defined(__DAV_CUBE__)` and its
  `#endif` (`TMUL` there is address arithmetic and is expected).
- **`pl.spmd` takes a Var, not a call.** Bind `pl.tensor.dim(...)` to a name first.
- **A script-entry file must put the repository root first on `sys.path`**, or
  `from golden import run` resolves to this directory's own `golden.py`.

## How this directory is organised for parallel work

Every operator has one ownership file. Each file carries the exact maths in its
module docstring, a `golden_*` Torch reference, and a typed `@pl.jit.inline` ABI
whose body raises `NotImplementedError` until its owner lands it. Shapes come
from `config.py`, so two engineers working on neighbouring operators agree on the
interface without talking to each other.

A file has **no script-entry guard until its golden is real**. The a2a3 daily CI
selects a case purely by grepping for that guard, so adding one to a file whose
golden still raises would put an unimplementable operator into the sweep. Today
`config.py`, `golden.py`, `quantization.py`, `metadata.py`, `mhc.py` and all five
stream-B MLA files carry real references, and eight of them run as CI cases:

```bash
source .venv/bin/activate-pypto
python models/glm5_3_flash/golden.py              # [GOLDEN] PASS norms / swiglu / moe_gate
python models/glm5_3_flash/quantization.py        # [GOLDEN] PASS quantization
python models/glm5_3_flash/mhc.py                 # [GOLDEN] PASS mhc
python models/glm5_3_flash/mla_prolog.py          # [GOLDEN] PASS mla_prolog + prolog/absorb
python models/glm5_3_flash/mla_cache.py           # [GOLDEN] PASS mla_cache + device scatter
python models/glm5_3_flash/mla_epilog.py          # [GOLDEN] PASS mla_epilog + both paths
python models/glm5_3_flash/prefill_sparse_attn.py # [GOLDEN] PASS + device sparse prefill
python models/glm5_3_flash/decode_sparse_attn.py  # [GOLDEN] PASS + device sparse decode
```

`models/deepseek_v4_1_flash/mhc.py` is the worked example of a finished item: a
golden, the four kernel bodies, a `@pl.jit` test entry, `build_mhc_tensor_specs`
and a precision comparison, in 559 lines. Read it for the shape, not for the
kernels — it is tagged `# ci: a5` and `# ci: no-sim`, and it even splits the
spelling of the entry sentinel (`_SCRIPT_ENTRY_POINT = "__" + "main__"`) so the
a2a3 sweep cannot pick it up. That idiom is the escape hatch if this directory ever
needs a file that is runnable by hand but must stay out of the a2a3 job.

For kernels you actually intend to run here, prefer the untagged a2a3 files:
`models/deepseek_v4_flash_mtp/` has implemented `hc_pre.py`, `hc_post.py` and
`hc_head.py`, `gate.py`, `expert_routed.py`, `expert_shared.py`, `decode_indexer.py`,
`prefill_indexer.py` and `decode_sparse_attn_csa.py`, all of which the a2a3 daily
sweep runs today.
## Five streams, five owners

The work divides into five self-contained streams. Each owns a disjoint set of
files, so two owners never edit the same file, and they meet only at the ABIs that
`config.py` and the `@pl.jit.inline` signatures already fix. Stream E owns those
contracts and publishes them first.


| Stream | Owns | Files | Items | Days | Long pole |
| --- | --- | ---: | ---: | ---: | --- |
| **A** | KDA linear attention — 34 of the 45 backbone layers | 5 | 10 | 50-82 | `kda_chunk_delta_prefill` — the chunked delta rule, 20-32 days on its own |
| **B** | NoPE MLA sparse attention — 11 sparse layers + the MTP layer | 5 | 6 | 45-72 | `mla_sparse_attn_prefill` — gather-and-attend over 2051 selected rows, 15-25 days |
| **C** | kpool DSA indexer and mHC — 12 indexer instances; mHC on all 45 layers | 4 | 11 | 38-60 | `indexer_cache` — it blocks five downstream items, so land it in week one |
| **D** | MoE, primitives, output head, MTP — 43 sparse layers, 3 dense layers, the draft step | 12 | 14 | 61-100 | `moe_dispatch_ep16` — the 16-rank all-to-all, 14-22 days |
| **E** | Runtime, weight loading, composition — the whole model | 9 | 6 | 76-120 | `layer_composition` — 40-60 days, and by definition it lands last |
| *(staged)* | Vision tower — prefill of multimodal requests only | 5 | 6 | 23-35 | `vision_attention` — non-causal varlen attention, no donor in this repo |

Phase one is streams A-E: **269-434 engineer-days** across five owners, before any tuning. The vision tower is staged out — the A3 recipe serves this checkpoint text-only unless `--limit-mm-per-prompt` is set, and the tower is 0.53 G parameters against the text model's 320 G.

### Stream A — KDA linear attention

*34 of the 45 backbone layers · 5 files · 10 work items · 50-82 engineer-days*

**Files.** `kda_projection.py`, `kda_conv.py`, `prefill_kda.py`, `decode_kda.py`, `kda_output.py`

**Where to start.** No donor in pypto-lib. The `pl`-adjacent reference is the tuned Ascend910B3 `chunk_kda_impl.py` / `fused_recurrent_kda_impl.py` in cann-recipes, plus its design doc.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `kda_chunk_delta_prefill` | SPLIT | XL | 20-32 | `kda_qkv_proj` |
| `kda_recurrent_decode` | ADAPT | L | 8-13 | — |
| `kda_conv1d_prefill` | NEW | M | 4-7 | `kda_qkv_proj`, `kda_conv1d_decode` |
| `kda_out_proj` | ADAPT | M | 4-7 | `mhc_post` |
| `kda_conv1d_decode` | NEW | M | 3-5 | `kda_state_cache`, `kda_qkv_proj` |
| `kda_state_cache` | NEW | M | 3-5 | — |
| `kda_gates` | NEW | S | 2-4 | `kda_qkv_proj` |
| `kda_qkv_proj` | ADAPT | S | 2-3 | — |
| `kda_l2norm_qk` | ADAPT | S | 1.5-3 | — |
| `kda_out_gated_norm` | ADAPT | S | 2-3 | — |

### Stream B — NoPE MLA sparse attention

*11 sparse layers + the MTP layer · 5 files · 6 work items · 45-72 engineer-days*

**Files.** `mla_prolog.py`, `mla_cache.py`, `prefill_sparse_attn.py`, `decode_sparse_attn.py`, `mla_epilog.py`

**Where to start.** `deepseek_v4_flash_mtp/qkv_proj_rope.py`, `prefill_sparse_attn.py` and `decode_sparse_attn_csa.py`. Same 512-wide KV row; delete the sliding window, the ratio-4 slot rewrite, the attention sink and all of the RoPE.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `mla_sparse_attn_prefill` | ADAPT | XL | 15-25 | — |
| `mla_sparse_attn_decode` | ADAPT | L | 9-14 | — |
| `mla_epilog_o_proj` | SPLIT | L | 8-13 | — |
| `mla_prolog_nope` | ADAPT | M | 6-9 | — |
| `mla_kv_absorb` | ADAPT | M | 4-6 | — |
| `mla_latent_cache` | ADAPT | S | 3-5 | — |

### Stream C — kpool DSA indexer and mHC

*12 indexer instances; mHC on all 45 layers · 4 files · 11 work items · 38-60 engineer-days*

**Files.** `prefill_indexer.py`, `decode_indexer.py`, `indexer_cache.py`, `mhc.py`

**Where to start.** `deepseek_v4_flash_mtp/{prefill,decode}_indexer.py` for the selection pipeline (its `_cp_topk512_query` is the same K over the same candidate cap), and `hc_{pre,post,head}.py` for mHC. The k-pooling stage has no donor anywhere.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `indexer_cache` | SPLIT | L | 7-10 | — |
| `indexer_score` | ADAPT | L | 6-9 | `indexer_cache` |
| `indexer_kpool_compress` | ADAPT | M | 5-8 | `indexer_cache` |
| `indexer_topk512` | ADAPT | M | 4-6 | `indexer_score`, `indexer_cache` |
| `indexer_qk_proj` | NEW | M | 3-5 | — |
| `indexer_share_mtp` | NEW | M | 3-5 | `indexer_score`, `indexer_cache` |
| `mhc_pre` | ADAPT | M | 3-5 | `mhc_post` |
| `mhc_mixes` | ADAPT | S | 2-4 | — |
| `indexer_expand_tail` | NEW | S | 2-3 | — |
| `mhc_post` | ADAPT | S | 1.5-3 | `mhc_pre`, `mhc_mixes` |
| `mhc_head` | ADAPT | S | 1-2 | `mhc_post` |

### Stream D — MoE, primitives, output head, MTP

*43 sparse layers, 3 dense layers, the draft step · 12 files · 14 work items · 61-100 engineer-days*

**Files.** `gate.py`, `expert_routed.py`, `expert_shared.py`, `dense_mlp.py`, `(new) prefill_moe.py`, `(new) decode_moe.py`, `rmsnorm.py`, `quantization.py`, `lookup_embedding.py`, `lm_head.py`, `mtp_projection.py`, `(new) decode_mtp.py`

**Where to start.** Almost every file has a same-named donor in `deepseek_v4_flash_mtp` that the a2a3 sweep already runs. This is the most port-like stream in the model.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `moe_dispatch_ep16` | SPLIT | XL | 14-22 | `moe_gate` |
| `mtp_draft_loop` | SPLIT | L | 12-18 | `indexer_score`, `indexer_cache` |
| `moe_expert_routed_w8a8` | ADAPT | M | 5-8 | `moe_dispatch_ep16`, `moe_gate`, `moe_combine_ep16` |
| `moe_combine_ep16` | ADAPT | L | 5-8 | `moe_dispatch_ep16`, `moe_gate`, `moe_expert_routed_w8a8` |
| `lm_head_tp16` | ADAPT | M | 4-7 | `mhc_head` |
| `dense_mlp_w8a8` | ADAPT | M | 4-6 | `moe_shared_expert_w8a8` |
| `moe_gate` | ADAPT | M | 3-5 | — |
| `moe_shared_expert_w8a8` | ADAPT | S | 3-5 | `moe_gate`, `w8a8_weight_loader` |
| `embed_tokens` | ADAPT | M | 3-5 | — |
| `swiglu_clamp_quant` | ADAPT | S | 2-4 | — |
| `mtp_eh_proj` | ADAPT | S | 2-4 | `rms_norm` |
| `w8a8_linear` | SPLIT | S | 1.5-3 | — |
| `rms_norm` | ADAPT | S | 1.5-2.5 | — |
| `quant_per_token_int8` | ADAPT | S | 1-2 | `moe_gate` |

### Stream E — Runtime, weight loading, composition

*the whole model · 9 files · 6 work items · 76-120 engineer-days*

**Files.** `config.py`, `metadata.py`, `(new) weights.py`, `(new) cache_manager.py`, `attention_tp.py`, `(new) prefill_layer.py`, `(new) decode_layer.py`, `(new) prefill_fwd.py`, `(new) decode_fwd.py`

**Where to start.** `deepseek_v4_flash_mtp/{prefill,decode}_{layer,fwd}.py`, about 8,600 lines of donor. The 3:1 KDA/DSA schedule replaces its SWA/CSA/HCA schedule.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `layer_composition` | SPLIT | XL | 40-60 | `mhc_head`, `mhc_post`, `mhc_mixes` |
| `tp16_collectives` | SPLIT | L | 15-27 | `forward_metadata` |
| `w8a8_weight_loader` | SPLIT | L | 8-12 | `indexer_score` |
| `hybrid_kv_cache` | ADAPT | L | 6-9 | `indexer_cache` |
| `forward_metadata` | ADAPT | M | 5-8 | — |
| `graph_capture` | ADAPT | S | 2-4 | — |

### Stream V — Vision tower (staged)

*prefill of multimodal requests only · 5 files · 6 work items · 23-35 engineer-days*

**Files.** `vision/patch_embed.py`, `vision/attention.py`, `vision/mlp.py`, `vision/merger.py`, `vision/fusion.py`

**Where to start.** No donor in this repo.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `vision_attention` | NEW | L | 8-12 | `vision_patch_embed` |
| `vision_mlp` | SPLIT | M | 5-7 | `vision_patch_embed`, `vision_attention` |
| `vision_rope_qkv_norm` | ADAPT | M | 4-6 | `vision_patch_embed`, `vision_attention` |
| `vision_merger` | NEW | M | 3-5 | `vision_patch_embed`, `vision_attention`, `vision_mlp` |
| `mm_fusion` | NEW | S | 2-3 | `vision_merger` |
| `vision_patch_embed` | ADAPT | S | 1-2 | — |

## Sequencing

Three items block the most downstream work and should go out in week one:
`indexer_cache` (blocks five), `moe_gate` (blocks five) and `kda_qkv_proj` (blocks
four). Stream E's contract work — `forward_metadata`, `w8a8_weight_loader`,
`tp16_collectives`, `hybrid_kv_cache` — is the other week-one batch, because
streams A through D cannot run anything on device without it.

That gives stream E a bookend shape: about 36-60 days of shared contracts up front,
then `layer_composition` at the end. It is the integration role, and it is also the
natural reviewer for the other four.

## Where each file's donor lives

`models/deepseek_v4_flash_mtp/` is the a2a3 W8A8 sibling, and almost every GLM file
has a same-shaped counterpart there that the a2a3 daily sweep already runs. The
port is, in one sentence: take that model, delete the compressor / sliding-window /
RoPE machinery, and add the KDA family and the kpool indexer.

| GLM file | a2a3 donor | What changes |
| --- | --- | --- |
| `mhc.py` | `hc_pre.py`, `hc_post.py`, `hc_head.py` | FP32 stream becomes BF16; the learned head becomes an unweighted mean |
| `rmsnorm.py` | `rmsnorm.py` | Add the INT8-emitting variant for the three dense layers |
| `gate.py` | `gate.py` | `sqrt(softplus(.))` becomes `sigmoid`; `SCORE_PAD` 256 to 512; drop hash routing and the group mask |
| `expert_routed.py`, `expert_shared.py` | same names | 288/8 instead of 256/6, `moe_intermediate_size` 2048 |
| `lookup_embedding.py`, `lm_head.py` | same names | Vocabulary 154,880; the embedding also seeds the four HC streams |
| `mtp_projection.py` | `mtp_projection.py` | Same `enorm` / `hnorm` / `eh_proj` shape |
| `mla_prolog.py` | `qkv_proj_rope.py` | Delete the whole RoPE half; add the `kv_b_proj` absorption |
| `prefill_sparse_attn.py`, `decode_sparse_attn.py` | `prefill_sparse_attn.py`, `decode_sparse_attn_csa.py` | Delete the sliding-window gather, the ratio-4 slot rewrite, the attention sink and the inverse RoPE |
| `prefill_indexer.py`, `decode_indexer.py` | same names | Delete the INT8/Hadamard half and the compressor; add the k-pooling stage |
| `(new) prefill_moe.py`, `decode_moe.py` | same names | EP16 instead of EP8; 18 local experts |
| `(new) prefill_fwd.py`, `decode_fwd.py` | same names | The 3:1 KDA/DSA schedule replaces the SWA/CSA/HCA schedule |
| `kda_*.py`, the k-pooling stage | **no donor in this repo** | See the prior-art table above |

## What already exists in the directory

| File | State |
| --- | --- |
| `config.py` | Complete. Layer schedule, derived per-rank shapes, TP/EP validation, W8A8 metadata |
| `golden.py` | Complete for the shared transforms: RMSNorm, gated RMSNorm, L2 norm, clamped SwiGLU, the expert body, the sigmoid `noaux_tc` router, and the four mHC references |
| `quantization.py` | Complete. Per-channel and per-token INT8 with the repo's exact rounding, the dynamic W8A8 linear, and the fused dequant-SwiGLU-requant epilogue |
| `metadata.py` | Complete. Packed-batch lowering, the three distinct slot mappings (latent, per-token indexer state, pooled state), TP token ownership, and the indexer's pool and tail derivation |
| `_golden_smoke.py` | Complete. Deterministic CPU fixtures behind every golden that exists |
| `mla_prolog.py` | Goldens, the prolog body (three `b_trans` matmuls and two rms-norm scopes, no split-K yet), the per-token `absorb_query` body, and a device test entry per case. `absorb_output` is weight-time and belongs to the loader. Passes `a2a3sim` and a2a3 |
| `mla_cache.py` | Golden, kernel body (`pl.spmd` scatter, one block per row), device test entry and specs. Passes `a2a3sim` and a2a3 |
| `mla_epilog.py` | Goldens, both kernel bodies (row-parallel `b_trans` projection over D, FP32 partial sum for the TP16 all-reduce) and a device test entry per path. Passes `a2a3sim` and a2a3 |
| `prefill_sparse_attn.py` | Golden, kernel body (one item per token, `pl.yield_`-carried online softmax, value expansion after the block loop) and a device test entry. **Attends in latent space**: absorbing keeps the two expansions out of the per-token block loop, at the cost of two BF16 roundings against the expanded reference. An all-padding block is filled by one wide gather and merged with `beta` = 0 rather than skipped, so no branch carries the online-softmax state. Passes `a2a3sim` and a2a3 |
| `decode_sparse_attn.py` | Golden, kernel body (24 lanes over (token, sparse block), L1 gather, flash partials, alpha/beta merge, empty-block skip) and a device test entry. Assumes a front-packed index list. Passes `a2a3sim` and a2a3 |
| `attention_tp.py` | ABI only, but it is the one shared consumer: `kda_output`, both `mla_epilog` entries and `dense_mlp` all emit an FP32 row-parallel partial, and this is what adds the 16 of them |
| everything else | ABI and docstring only; the golden and the kernel body are the assignment |

## Open questions an owner will hit

- **Whether the MTP layer owes `rot.weight`.** `is_rot_used: true` and the tensor's
  presence in the index argue yes; `Glm5NextMTP` dropping it argues no, or argues
  the reference port has a gap. Stream D settles it before implementing MTP.
- **How the indexer cache is laid out.** vLLM Ascend splits it into an index-K
  cache and a separate FP32 pooled-state page class with a page of four tokens;
  storing the raw row instead is simpler but larger. Stream C decides, stream E's
  cache manager implements.
- **Whether the indexer's index list is front packed.** `decode_sparse_attn` skips a sparse block whose lane 0 is `-1`, which turns a 40-selection request from 2,176 gathered rows into 128. Stream C owns the emission order; if a list can interleave `-1` with live selections the skip has to go, or the indexer has to compact.
- **Whether the MLA decode path absorbs.** `kv_b_proj` is BF16 in the checkpoint,
  so folding it into an INT8 `o_proj` changes the numerics of the epilogue.
- **The cache budget.** At about 18 KB per token per rank the hybrid cache holds
  roughly 1.5 M tokens per node once weights are placed, so the recipe's
  `--max-model-len 133120` and `--max-num-seqs 32` cannot both be satisfied at
  full length.
