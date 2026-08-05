# DeepSeek

PyPTO-Lib contains split DeepSeek V3.2-EXP harnesses and two DeepSeek V4
kernel trees. V4 Flash is covered by the A2/A3 and simulator daily sweeps;
V4 Pro has a dedicated A5 device sweep.

## DeepSeek V3.2-EXP

The V3.2-EXP tree contains three runnable components:

| Entry | Scope | Declared platforms | Configured CI coverage |
| --- | --- | --- | --- |
| [decode_front.py](../../models/deepseek_v3_2/decode_front.py) | Fused decode front scopes | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [decode_back.py](../../models/deepseek_v3_2/decode_back.py) | Single-layer decode back | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [prefill_back.py](../../models/deepseek_v3_2/prefill_back.py) | Reduced single-layer prefill back fixture | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |

These are component harnesses rather than a complete prefill/decode model
contract. The prefill-back fixture is deliberately reduced to batch 1 and
sequence length 128 in source.

```bash
python models/deepseek_v3_2/decode_front.py -p a2a3sim
```

## DeepSeek V4

### Variants and limits

| Variant | Directory | Runnable CLIs | Primary CI target | Kernel-program sequence budget |
| --- | --- | ---: | --- | ---: |
| Flash | [deepseek_v4_flash_mtp](../../models/deepseek_v4_flash_mtp/) | 37 | A2/A3; A2/A3 sim and A5 sim for non-device-only cases | 16,384 |
| Pro | [deepseek_v4_pro](../../models/deepseek_v4_pro/) | 35 | Dedicated A5 daily job | 16,384 |

The Pro model configuration retains its architectural one-million-position
value, but the tracked Pro kernel programs deliberately use a 16,384-position
test budget so paged-cache fixtures and host goldens remain practical.

Both tracked V4 presets describe FP8 model data and FP4 routed-expert weights.
That configuration metadata is not proof that every standalone harness loads
or validates real quantized checkpoint data; each harness's tensor specs and
golden function remain the authority.

A directory suffix names the speculative-decoding method the tree's token
budget is built for. `config.py` is a per-directory singleton — every kernel
reads `DECODE_BATCH` / `DECODE_SEQ` from it by bare sibling import — so a
method with a different speculative-token count needs its own tree. Both
current trees carry `DECODE_SEQ = 2`, the MTP shape of one speculative token
verified against the previous one. Flash is `deepseek_v4_flash_mtp` because a
DSpark sibling with a larger budget is planned; Pro has no suffix because it
has nothing to disambiguate from.

Quantization is not part of the directory name. Both trees consume INT8
weights today — the Pro routed-expert kernels included, where
`gen_routed_weight` in
[expert_routed.py](../../models/deepseek_v4_pro/expert_routed.py)
re-quantizes off the MXFP4 grid rather than implementing native
MXFP8-MXFP4. Add a quantization suffix only when one model, variant, and
method genuinely ship in two quantizations.

### Entry-point classes

The Flash and Pro trees share the same broad organization:

| Class | Representative entries | Execution status |
| --- | --- | --- |
| Full forward | `decode_fwd.py`, `prefill_fwd.py` | Multi-layer, multi-card, device-only; default CI marker borrows 2 cards |
| Layer harnesses | `decode_layer.py`, `prefill_layer.py` | Attention followed by distributed MoE; default EP2 |
| MTP harnesses | `decode_mtp.py`, `prefill_mtp.py` | Multi-token-prediction composition; 2-card marker |
| Attention orchestration | `*_attention_csa.py`, `*_attention_hca.py`, `*_attention_swa.py` | Runnable single-device component harnesses |
| Sparse attention and cache | `*_sparse_attn*.py`, compressor and indexer files | Runnable component harnesses |
| MoE and output | `moe.py`, `gate.py`, `expert_*.py`, `lm_head.py` | Mix of single-device components and 2-card distributed harnesses |
| Shared transforms | `qkv_proj_rope.py`, `rmsnorm.py`, `hc_*.py`, `mtp_projection.py` | Runnable component harnesses |
| Imported support | `config.py`, metadata helpers, RoPE table helpers | Library modules without standalone CLIs |

There are no tracked `*_draft.py` files in either V4 directory. A runnable
`__main__` block still does not make a component a full model forward; the
classification above reflects what the harness actually invokes.

### Distributed coverage

The V4 full-forward, layer, MTP, MoE, and LM-head files that carry
`# ci: devices=2` are tested at their default two-rank configuration. Several
CLIs accept `--ep 2`, `--ep 4`, or `--ep 8`, but the daily workflow does not
currently add active EP4 or EP8 per-file runs. A commented command example in
the workflow is not test coverage.

Changes under the Flash directory also trigger a shared eight-card
`pypto-serving` accuracy job on pull requests. Pro does not: it is the A5
variant and no serving deployment consumes it. The job verifies the external
serving integration path; it does not change the two-card status of individual
kernel harnesses.

### Platform coverage

V4 platform declarations are entry-specific:

- many component parsers accept `a2a3`, `a2a3sim`, `a5`, and `a5sim`;
- the full-forward parsers accept real-device `a2a3` and `a5` only;
- `no-sim` markers exclude full forwards and prefill MTP from simulator sweeps.

The daily workflow runs all 37 Flash CLIs on A2/A3 and the 34 non-`no-sim`
Flash CLIs on both simulators. V4 Pro is excluded from those sweeps and is
instead scheduled as 35 CLIs on an A5 runner. The aggregate daily summary does
not depend on the A5 job, and the workflow notes that it can remain queued
until an A5 runner is registered. Treat A5 as configured coverage and inspect
the job result before citing a revision as verified.

### Run V4 harnesses

Run a single-device Flash attention component on the simulator:

```bash
python models/deepseek_v4_flash_mtp/prefill_csa.py -p a2a3sim
```

Run the Flash decode-layer harness at its default EP2 world size:

```bash
python models/deepseek_v4_flash_mtp/decode_layer.py \
  -p a2a3 --ep 2 -d 0,1
```

Run the equivalent Pro layer on two A5 devices:

```bash
python models/deepseek_v4_pro/decode_layer.py \
  -p a5 --ep 2 -d 0,1
```

Full-forward cases are substantially heavier than component harnesses. Start
with a layer or attention case and inspect the selected script's `--help`
before increasing token counts, cache sizes, or world size.
