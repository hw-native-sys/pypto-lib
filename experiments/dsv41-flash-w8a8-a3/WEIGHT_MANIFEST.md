# Weight mapping

Spec revision: 1 (2026-11-02). Mapping executed by the implementation agent with
`models/deepseek_v4_1_flash_a3_w8a8/weight_map.py` against the read-only
checkpoint; frozen output `weight_map.json` in this directory.

Checkpoint identity: `DeepSeek-V4.1-Flash-w8a8` (read-only),
`/mnt/data/models/DeepSeek-V4.1-Flash-w8a8`. On disk: 270 quant shards +
4 standalone Engram files + index + descriptions + rotation map + shipped
encoding implementation; 742.7 GiB total. Aurora manifest
`aurora_export_manifest_v1` (completed 2026-09-10) records 272 shards and
export-integrity-only scope.

## Completeness verification (executed 2026-11-02, observer)

- `quant_model_weights.safetensors.index.json` maps 143,496 tensors to 274
  files; every referenced file exists.
- Header scan of all 274 on-disk safetensors: 143,496 of 143,496 expected
  tensor names present, 0 missing, 0 unexpected.
- Aurora manifest records shards 00271/00272 as absent from disk; their
  recorded bytes equal the 4 standalone Engram files; the content was
  relocated, the checkpoint is complete.

## Mapping result (executed 2026-11-02, implementation agent)

Command:

```bash
python models/deepseek_v4_1_flash_a3_w8a8/weight_map.py \
  --output experiments/dsv41-flash-w8a8-a3/weight_map.json
```

Actual summary (`exit 0`):

| Field | Value |
| --- | --- |
| Tensors in index | 143,496 |
| Included (text backbone + Engram) | 139,664 |
| Excluded (vision/aligner/image/MTP) | 3,832 |
| Included bytes | 779,156,733,952 (725.7 GiB) |
| Excluded bytes | 18,256,696,712 (17.00 GiB) |
| Routed-expert tensors | 138,240 (544,777,175,040 B) |
| Routed-expert bytes per EP16 rank | 34,048,573,440 (31.70 GiB) |
| Device HBM bytes per rank (mapping-derived) | 39,455,799,232 (36.75 GiB) |
| Engram host table bytes | 2 x (98.3e9 + 12.3e9) = 206.59 GiB |
| Unmapped included tensors | 0 |
| Index names without a quant-description tag | 0 |

The mapping-derived per-rank figure is more precise than the earlier uniform
`dense_total / TP4` estimate (34.63 GiB): `embed/head`, `wq_b`, `wo_a`, `wo_b`
and `indexer.wq_b` are TP4-sharded, while `wq_a`, `wkv`, norms, compressors,
gate, HC and shared experts are replicated per rank and routed experts are
EP16-sharded. Both estimates fit the 57.6 GiB usable budget; the refinement is
recorded rather than replacing the measured observer figure.

### Quantization offset check

47,498 tensors carry `weight_offset`. A 201-tensor stride sample across
attention and expert families was read and all values are zero. The loader
still applies `w = (q - offset) * scale`; the mapping transform folds the
offset at load time. The device GEMM therefore uses `acc * activation_scale *
weight_scale` with no residual offset term.

## Mapping table (text backbone, TP4/EP16)

Shapes are logical `[N, K]` for W8A8 (checkpoint order) and the model argument
transposes to `[K, N]` before NZ packing. `scale`/`offset` are `[N, 1]` F32.

| Checkpoint family (per layer unless stated) | Source shape / dtype | Transform | Rank-local shape / dtype | Destination argument |
| --- | --- | --- | --- | --- |
| `embed.weight` | [129280, 5120] BF16 | vocab TP4 slice | [32320, 5120] BF16 | `embed_weight` |
| `head.weight` | [129280, 5120] BF16 | vocab TP4 slice | [32320, 5120] BF16 | `lm_head_weight` |
| `norm.weight` | [5120] BF16 | keep | [5120] BF16 | `final_norm` |
| `layers.N.attn_norm.weight` | [5120] BF16 | keep | [5120] BF16 | `layers.N.attn_norm` |
| `layers.N.ffn_norm.weight` | [5120] BF16 | keep | [5120] BF16 | `layers.N.ffn_norm` |
| `layers.N.attn.wq_a.weight` + scale/offset | [1280, 5120] I8, [1280,1] F32 x2 | transpose -> NZ; offset folded | [5120,1280] I8 NZ + [1280] F32 | `layers.N.wq_a`, `_scale` |
| `layers.N.attn.wq_b.weight` + scale/offset | [32768, 1280] I8, F32 x2 | head TP4 slice, transpose -> NZ | [1280, 8192] I8 NZ + [8192] F32 | `layers.N.wq_b`, `_scale` |
| `layers.N.attn.wkv.weight` + scale/offset | [512, 5120] I8, F32 x2 | transpose -> NZ | [5120, 512] I8 NZ + [512] F32 | `layers.N.wkv`, `_scale` |
| `layers.N.attn.q_norm.weight` | [1280] BF16 | keep | [1280] BF16 | `layers.N.q_norm` |
| `layers.N.attn.kv_norm.weight` | [512] BF16 | keep | [512] BF16 | `layers.N.kv_norm` |
| `layers.N.attn.attn_sink` | [64] F32 | head TP4 slice | [16] F32 | `layers.N.attn_sink` |
| `layers.N.attn.wo_a.weight` | [8192, 4096] BF16 | output-group TP4 slice | [2048, 4096] BF16 | `layers.N.wo_a` |
| `layers.N.attn.wo_b.weight` | [5120, 8192] BF16 | output-group TP4 slice | [5120, 2048] BF16 | `layers.N.wo_b` |
| `layers.N.attn.compressor.wkv.weight` | [512, 5120] BF16 | keep | same | `layers.N.compressor_wkv` |
| `layers.N.attn.compressor.wgate.weight` | [512, 5120] BF16 | keep | same | `layers.N.compressor_wgate` |
| `layers.N.attn.compressor.norm.weight` | [512] BF16 | keep | same | `layers.N.compressor_norm` |
| `layers.N.attn.indexer.wq_b.weight` + scale/offset | [4096, 1280] I8, F32 x2 | index-head TP4 slice (8 heads), transpose -> NZ | [1280, 1024] I8 NZ + [1024] F32 | `layers.N.indexer_wq_b`, `_scale` |
| `layers.N.attn.indexer.wk.weight` | [128, 512] BF16 | keep | same | `layers.N.indexer_wk` |
| `layers.N.attn.indexer.weights_proj.weight` | [32, 5120] BF16 | keep | same | `layers.N.indexer_weights_proj` |
| `layers.N.attn.indexer.k_norm.weight` | [128] BF16 | keep | same | `layers.N.indexer_k_norm` |
| `layers.N.ffn.gate.weight` | [384, 5120] BF16 | keep (replicated) | same | `layers.N.gate_weight` |
| `layers.N.ffn.gate.bias` | [384] F32 | keep | same | `layers.N.gate_bias` |
| `layers.N.ffn.gate.bias_vl` | [384] F32 | keep (unused in text-only) | same | `layers.N.gate_bias_vl` |
| `layers.N.hc_attn_fn` / `hc_ffn_fn` | [24, 20480] F32 | keep | same | `layers.N.hc_attn_fn` / `hc_ffn_fn` |
| `layers.N.hc_attn_scale` / `hc_ffn_scale` | [3] F32 | keep | same | `layers.N.hc_*_scale` |
| `layers.N.hc_attn_base` / `hc_ffn_base` | [24] F32 | keep | same | `layers.N.hc_*_base` |
| `layers.N.ffn.experts.E.w{1,3}.weight` + scale/offset | [2304, 5120] I8, F32 x2 | EP16 expert placement (`E // 24 = rank`, `E % 24 = local`), transpose -> NZ | [5120, 2304] I8 NZ + [2304] F32 | `layers.N.experts[local]` |
| `layers.N.ffn.experts.E.w2.weight` + scale/offset | [5120, 2304] I8, F32 x2 | EP16, transpose -> NZ | [2304, 5120] I8 NZ + [5120] F32 | `layers.N.experts[local]` |
| `layers.N.ffn.shared_experts.w{1,3,2}` + scale/offset | [2304,5120] / [5120,2304] I8, F32 x2 | replicated per rank, transpose -> NZ | same logical | `layers.N.shared_w*` |
| `layers.1/14.engram.q_weight`, `k_weight` | [4, 5120] BF16 | keep, fold `q*k` per channel on host | [4, 5120] F32 | `engram_qk[slot]` |
| `layers.1/14.engram.wkv.weight` | [25600, 6144] BF16 | keep (key 4x5120 + value 5120) | same | `engram_wkv[slot]` |
| `layers.1/14.engram.embed.weight` | [384006168/384016682, 256] I8 | int8_sym group-32, host offload | host table | `HostEngram` lookup row |
| `layers.1/14.engram.embed.scale` | [N, 8] F32 | group-32 scales, host offload | host table | `HostEngram` dequant |
| `optional/quarot.safetensors:global_rotation` | [5120,5120] floating point | verify repeated diagonal block32 and zero off-block entries; retain stored orientation | [32,32] F32, replicated per rank, shared across Engram layers | `engram_rotation` |

Engram resource binding (2026-11-06 implementation): `WeightLoader.engram_rotation`
loads and caches this bounded optional map; it does not materialize the model or
Engram embedding tables. Synthetic invalid structure/shape/contract/missing-map
controls and an actual map-only read validate the loader path. The observed
real block SHA256 is `68fef81121ad4d31c34d7659a684bff34d2fc1de23bc46201cf6cf35606fca1a`,
with max orthogonality error5.96e-8. q*k remains folded per original channel;
gate hidden is restored with R.T before using it. Wkv is loaded as-is: its V
rows already contain the export rotation and must not be rotated again.
Independent cohort acceptance is separate from this implementation record.

## Excluded tensors (with reasons)

| Reason | Count | Bytes | Examples |
| --- | --- | --- | --- |
| Vision tower (text-only phase 1) | 259 | 823,685,120 | `vision.blocks.0.attn.wo.weight` |
| Aligner (text-only phase 1) | 4 | 146,821,120 | `aligner.w1.weight` |
| Image token tensors | 3 | 30,720 | `image_start`, `image_end`, `image_newline` |
| MTP/DSpark layers (excluded by spec) | 3,566 | 17,286,159,752 | `mtp.0.*` ... `mtp.2.*` |

MTP/DSpark include `mtp.2.markov_head`, `mtp.2.confidence_head`,
`mtp.2.main_proj`, `mtp.2.head`, and MTP attention/FFN. They remain unmapped by
design; the checkpoint files are untouched.

## Open findings

| Finding | Impact | Owner / remedy | Status |
| --- | --- | --- | --- |
| Checkpoint README prose says Engram tables are decoded to BF16 without scales; headers and `quant_model_description.json` say int8_sym group-32 + F32 scales, BF16 output | Loader and host-memory footprint use int8 + group scales | Treat headers/description as ground truth (frozen decision 7) | resolved 2026-11-02 |
| Aurora export includes QuaRot rotation map; the reference A5 code contains no rotation application (assumes the hidden stream is already rotated) | Real-weight generation must confirm whether `embed.weight` output needs `h_rot` transformation before layer 0 and how Engram `q/k/gate` bases compose | Resolved from `Aurora_best_practice.yaml` + tensor inspection: weights are stored pre-rotated and pre-norm-folded; load as-is (see contract below) | resolved 2026-11-02 |
| Mapping-derived per-rank 36.75 GiB vs observer uniform estimate 34.63 GiB | Both fit; the difference is replicated dense tensors (wq_a/wkv/gate/HC/shared experts) | Use the mapping-derived figure for allocation | resolved (refined) |


## QuaRot and norm-folding contract (resolved 2026-11-02)

Evidence:

1. `Aurora_best_practice.yaml` `spec.quarot` records the export-time
   transforms: `right_global` (`W <- W Q`) for `embed.weight`, `head.weight`,
   `wq_a`, `wkv`, compressor `wkv/wgate`, `indexer.weights_proj`, `gate`,
   expert/shared `w1/w3`, and `hc_*`; `left_global` (`W <- Q^T W`) for
   `wo_b`, expert/shared `w2`; `right_query` for `wq_b`/`indexer.wq_b`;
   `left_query` plus `right_global` for `wq_a`; `engram.fold_value_rotation:
   true`.
2. The shipped rotations are orthogonal: `global_rotation` [5120,5120] and
   `query_rotation` [1280,1280] have `max|Q Q^T - I| = 5.96e-8` and
   `1.19e-7` respectively.
3. `norm_folding` lists `norm.weight`, `layers.*.attn_norm.weight`,
   `layers.*.attn.q_norm.weight`, `layers.*.ffn_norm.weight` as folded into
   the following linear weights. Measured values are exactly `1.0` for every
   element of `layers.0.attn_norm.weight`, `layers.0.ffn_norm.weight`,
   `layers.0.attn.q_norm.weight`, and `norm.weight`; `kv_norm.weight` is not
   in the fold list and retains real non-unit values (min 0.0182, max 0.4922).
4. `engram.fold_value_rotation: true` and the README's Engram statement say
   the last 5120 rows of `engram.wkv` already have `Q^T Wv` folded; the
   residual update is `h_rot + gate * v_rot` with no extra rotation.

Decision: **the checkpoint stores backbone weights pre-rotated (global/query
basis) and norm-folded. The loader loads them as-is; no QuaRot or norm-folding
transform is applied at load or dequant time.** The activation stream after
`embed.weight` is already in the rotated basis, and the right/left transform
list preserves that basis through the stack; the head (`right_global`) emits
logits in the original basis. The unit stored norm weights may still be
applied (multiplication by exactly 1.0 is exact) or skipped; the model keeps
the norm call for interface fidelity. Engram `gate` input still needs
`h_rot @ Q.T` per the README only for the original-basis `k_weight`/`q_weight`
product; that is a host-side Engram detail recorded for the Engram module and
does not change backbone loading.
