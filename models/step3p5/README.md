# Step3p5 — PyPTO multi-card (TP=8 + EP=8) implementation

This directory implements the **step3p5** language model on the pypto
programming framework, targeting an Ascend NPU node. The full
deployment is single-node, 8 cards, one process per card, with both
tensor-parallel (TP=8) and expert-parallel (EP=8) groups co-located on
the same world. The target checkpoint is the BF16 HuggingFace artefact
at `/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16/`.

## What landed

### Kernel modules (Phases 1–9, 15 files)

| File | Role |
|---|---|
| `config.py` | Source of truth: per-layer tables, TP/EP topology constants, helpers (`is_full_attention`, `is_moe_layer`, `ep_expert_owner`, …). |
| `_ops.py` | Cross-kernel inline helpers: `tp_all_reduce` (ring reduce-scatter + AllGather) and `zero_centered_rmsnorm_apply`. |
| `collectives.py` | TP/EP collective primitives shared by attention / MoE / lm_head. |
| `attention_full.py` | Full-attention decode (64 heads → 8/rank, partial-rotary=0.5, theta=5e6, llama3 yarn rope). |
| `attention_swa.py` | Sliding-window decode (96 heads → 12/rank, window=512, full-rotary, theta=1e4). |
| `gate.py` | FP32 sigmoid + router-bias top-8 + renormalize + ×3.0 scaling factor. |
| `dispatch.py` | EP all-to-all token scatter onto local expert windows. |
| `expert_routed.py` | 36 local routed experts × `MOE_INTERMEDIATE=1280` SwiGLU FFN. |
| `expert_shared.py` | TP-sliced 1280-d shared expert with per-layer activation limits. |
| `combine.py` | EP all-to-all weighted gather of expert outputs back to caller. |
| `moe.py` | Composes gate/dispatch/expert_routed/expert_shared/combine into `EpTpMoE_*` programs. |
| `decode_layer.py` | Eight per-layer `@pl.program` specialisations (full/swa × dense/MoE×(silu/swiglu7/swiglu16)) + `select_decode_layer(layer_idx)` dispatcher. |
| `decode_fwd.py` | Top-level decode orchestration (45 layers + rms_lm_head). |
| `rms_lm_head.py` | TP vocab-sliced final RMSNorm + LM head matmul. |
| `mtp.py` | 3 multi-token-prediction layers (eh_proj row-slice + SWA + dense + shared_head). |

### Phase 8 integration deliverables (this commit)

| File | Role |
|---|---|
| `weight_loader.py` | **Host-side weight loader.** Maps the HF safetensors checkpoint into per-rank pypto weight bundles. Exposes `load_step3p5_weights_for_rank(ckpt_dir, rank, tp_world_size=8)`, `verify_bundle_shapes(bundle, tp_world_size)`, `expected_shapes(tp_world_size)`, `build_compact_shape_table(...)`, and `build_synthetic_bundle(...)`. |
| `step3p5_decode.py` | **Top-level decode smoke entry.** CPU torch reference walking 45 main layers + 3 MTP layers using a per-rank weight slice. CLI mirrors the in-tree dense-GQA entries (`-p {a2a3,a2a3sim,a5,a5sim}`, `-d <device>`). |
| `step3p5_prefill.py` | **Top-level prefill smoke entry.** Sequence-major `[T, HIDDEN]` torch reference walking the same 45-layer stack. Same per-rank bundle + dispatcher; the `--no-smoke` real-NPU path is currently stubbed pending the rest of the Phase 6 prefill kernels. |
| `README.md` | This file. |

### Drafts kept for reference

* `single_layer_decode_full_draft.py` — Phase 2a single-card draft of layer 0 (full attention + dense MLP).
* `single_layer_decode_swa_draft.py` — Phase 2b single-card draft of layer 1 (sliding attention + dense MLP).

These are excluded from CI (`_draft.py` suffix) and remain in-tree as
the canonical single-card reference for the parametric attention work in
Phase 3.

## Topology

```
TP_WORLD_SIZE = 8
EP_WORLD_SIZE = 8
world_size    = 8           # single-node, one process per card
```

* Attention Q/K/V/O sharded by head count (`NUM_HEADS_FULL=64`,
  `NUM_HEADS_SWA=96`, `NUM_KV_HEADS=8` → 1 KV head per rank).
* `lm_head` and per-MTP `shared_head.output` sharded by vocab
  (`VOCAB_LOCAL=16112` per rank).
* Dense MLP / shared expert sharded by intermediate dim
  (`INTERMEDIATE_LOCAL=1408`, `SHARE_EXPERT_DIM_LOCAL=160`).
* Routed experts sharded by global expert id, 36 contiguous experts per
  card (`288 / 8 = 36`).
* MTP `eh_proj` row-sliced to `[HIDDEN_LOCAL=512, 2*HIDDEN=8192]` per
  card; the kernel writes the rank's partial into a zero-padded
  `[BATCH, HIDDEN]` buffer and runs `tp_all_reduce` to assemble the
  full output (the same "row-slice then tp_all_reduce as all-gather"
  trick used elsewhere on the residual stream).

## How to run

### CPU smoke (no NPU, no checkpoint required)

Decode:
```bash
cd <pypto-lib root>
python -m models.step3p5.step3p5_decode -p a2a3sim
```

Prefill:
```bash
python -m models.step3p5.step3p5_prefill -p a2a3sim -b 1 -s 128
```

Both build a compact synthetic per-rank bundle (the production-size
bundle is ~72 GB per rank and will not fit in host RAM), walk the
torch reference across all 45 + 3 layers, and print a pass-rate report.
The default threshold is `pass_rate >= 0.95`. The prefill entry also
asserts top-1 token agreement at the last position of each sequence.

### Smoke with real checkpoint slices

When the checkpoint share is mounted:

```bash
python -m models.step3p5.step3p5_decode -p a2a3sim --from-ckpt --rank 0
```

The loader materialises **only** rank 0's slice (it is `~36 B params *
2 bytes / 8 ranks ≈ 8–9 GB`); other ranks need their own invocation on
the box(es) hosting them. Falls back to synthetic if the share is not
reachable.

### Full 8-card deployment

```bash
# One process per card, with rank assigned by the launcher:
python -m models.step3p5.step3p5_decode -p a2a3 -d 0 --no-smoke
```

The `--no-smoke` path defers to `decode_fwd.Step3p5DecodeFwd`. As of
this commit the harness wiring (per-rank window allocator, KV-cache
build, RoPE table host build, multi-process orchestration) is not yet
landed and the entry raises a descriptive `NotImplementedError`
directing the operator to the smoke path. The kernel modules themselves
are TP/EP-aware and were validated layer-by-layer in Phases 3–9.

### How to load weights programmatically

```python
from models.step3p5.weight_loader import (
    load_step3p5_weights_for_rank,
    verify_bundle_shapes,
)

bundle = load_step3p5_weights_for_rank(
    ckpt_dir="/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16",
    rank=0,
    tp_world_size=8,
)
verify_bundle_shapes(bundle, tp_world_size=8)
# bundle["wq_full"].shape == (12, 4096, 1024)     # 12 full layers, Q_LOCAL=1024
# bundle["moe_w_gate_r"].shape == (42, 36, 4096, 1280)  # 36 experts/rank
# ...
```

The returned dict has 45 keys covering all replicated, TP-sliced, and
EP-sliced tensor categories — see the docstring at the top of
`weight_loader.py` for the complete layout.

## Migration plan status

| Phase | Topic | Status |
|---|---|---|
| 1 | `config.py` + `MIGRATION_PLAN.md` | done |
| 2a | Single-layer full draft (layer 0) | done (kept as `_draft.py`) |
| 2b | Single-layer SWA draft (layer 1) | done (kept as `_draft.py`) |
| 1.5 | Verify checkpoint shapes vs config | done |
| 3 | Parametric attention (`_ops`, `attention_full`, `attention_swa`, `decode_layer`) | done |
| 4 | MoE block (`gate`, `dispatch`, `expert_routed`, `expert_shared`, `combine`, `moe`) | done |
| 5 | `decode_fwd` (45 layers) + `rms_lm_head` | done |
| 6 | Prefill pipeline | in progress (sibling task) |
| 7 | MTP layers (`mtp.py`) | done |
| 8 | End-to-end integration + smoke + weight loader | done (this commit) |
| 9 | Convert entire impl to TP=8 + EP=8 in-place | done |

## Open items (deferred to first-NPU validation)

These items did not block Phase 8 acceptance but require live NPU time
to close out:

1. **`Q_HEAD_PAD_SWA = 24` NPU bounce.** The fa_fused path padding for
   SWA (96 heads → 12/rank × Q_PER_KV) sits at 24, the smallest multiple
   of 4 satisfying the cube constraints. There is no signal from CI
   that this trips on a2a3, but the in-tree tuning passes for the
   dense-GQA reference model used 16; first-NPU run should confirm the
   SWA kernels compile + run without re-tiling.
2. **EP all-to-all perf tuning.** `dispatch.py` and `combine.py` use
   conservative ring-buffer sizing (`LOCAL_RECV_MAX = 1024`,
   `BATCH * MOE_TOP_K = 128`); production swimlane / PMU reports will
   tell us whether to grow them.
3. **Real-NPU harness wiring.** `step3p5_decode.py --no-smoke` currently
   raises `NotImplementedError`. Wiring the per-rank window allocator,
   KV-cache build path, RoPE table host build, and multi-process
   launcher is the next deliverable after the smoke entry lands.
4. **Real-checkpoint loader validation.** The `weight_loader` slicing
   math is exercised by `verify_bundle_shapes` against the synthetic
   bundle; first checkpoint load on a host with the network share
   mounted should confirm the HF tensor-name mapping (`g_proj` head
   axis order, `moe.router_bias` FP32 dtype, MTP `shared_head.output`
   ckpt name) lines up with what's actually on disk.
5. **End-to-end pass-rate with real weights.** The smoke entry's
   threshold (`>= 0.95`) is a placeholder for the BF16 long-tail with
   the stub attention path; once the per-kernel attention is wired
   into the torch reference (instead of the identity-on-residual
   placeholder), the threshold should tighten.

## File summary

```
models/step3p5/
├── config.py                          # shapes + per-layer tables
├── _ops.py                            # inline helpers (tp_all_reduce, zc_rmsnorm)
├── collectives.py                     # TP/EP collective primitives
├── attention_full.py                  # full attention (64 heads → 8/rank)
├── attention_swa.py                   # sliding attention (96 heads → 12/rank)
├── gate.py                            # FP32 sigmoid + bias top-8
├── dispatch.py                        # EP a2a token scatter
├── expert_routed.py                   # 36 local routed experts
├── expert_shared.py                   # TP-sliced shared expert
├── combine.py                         # EP a2a weighted gather
├── moe.py                             # EpTpMoE_* program composition
├── decode_layer.py                    # 8 per-layer specialisations + dispatcher
├── decode_fwd.py                      # top-level decode @pl.program
├── rms_lm_head.py                     # final RMSNorm + vocab-sliced LM head
├── mtp.py                             # 3 MTP layers
├── prefill_qkv_proj_rope.py           # Phase 6: prefill QKV projection + RoPE
├── prefill_attention_full.py          # Phase 6: prefill full attention
├── weight_loader.py                   # Phase 8: HF ckpt → per-rank bundle
├── step3p5_decode.py                  # Phase 8: end-to-end decode smoke entry
├── step3p5_prefill.py                 # Phase 8: end-to-end prefill smoke entry
├── single_layer_decode_full_draft.py  # Phase 2a reference (not in CI)
├── single_layer_decode_swa_draft.py   # Phase 2b reference (not in CI)
├── MIGRATION_PLAN.md                  # Phase 1 plan doc
└── README.md                          # this file
```
