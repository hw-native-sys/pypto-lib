# F01 bounded repair: common MLA query equation

The common rank-local projection now computes
`qr = RMSNorm(BF16(Wq_a(quant(x))))`, followed by
`q = RoPE(BF16(Wq_b(quant(qr))))`. It retains the learned per-head
magnitude. The previous head RMSNorm is removed, and Wq_b dequantization
rounds to BF16 **before** the rotary arithmetic. KV normalization/rotation,
query-latent quantization and all public arguments retain their contracts.

Only `models/deepseek_v4_1_flash_a3_w8a8/mla_proj.py` changed among the
45 model sources. All twelve phase entries call this shared implementation
(C1A decode reuse delegates to shared decode reuse). Their goldens and the
layer/full-forward references already import `_golden_mla_proj`, so correcting
that owning golden propagates without duplicate implementations. The accepted
indexer width/tail, reindex consumption and padding comparator are byte-identical.

## Independent semantic grounding

Authority remains goal decisions 1–9, the W8A8 checkpoint and the agreed
vllm-ascend A3 form. Verified official archive SHA256
`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`
and byte equality of all three executed upstream source files against that
archive. Tag/revision: `v0.30.0rc2`,
`8fde6d4e335fe300e2b949aab0e31659cce0811d`.

- `vllm_ascend/attention/dsa_v41.py:320`: `_project_q` calls q_norm,
  Wq_b and partial RoPE directly; no head norm. `_project_kv:327`
  retains the KV norm. The declared but unused `q_norm_without_weight`
  member in `model.py` does not add an operation to this equation.
- `vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:104–135`:
  the ordinary/prepared linear paths preserve the input/output activation
  dtype and return `npu_quant_matmul(..., output_dtype=output_dtype)`.
  For the selected BF16 stream this is BF16 before `_apply_rotary`.
  WEIGHT_MANIFEST maps real Wq_b to INT8 plus FP32 per-channel scales.
- Upstream `tests/e2e/nightly/single_node/ops/singlecard_ops/`
  `test_inplace_partial_rotary_mul.py` supplies its own FP32 rotary CPU
  golden with BF16 narrowing, adjacent pairs and unchanged nonrotary lanes.

`check_semantics.py` AST-extracts and executes the original projection,
W8A8 dispatch and upstream rotary golden. It models NPU dynamic quant and
INT32 GEMM on CPU, without calling candidate quant/GEMM/normalization/rotary
goldens. RMSNorm is supplied as its mathematical BF16 equation. This is
independent semantic execution, **not** execution of original torch_npu
primitives on hardware or original full-model/checkpoint generation.

The permanent owning operator default now adds T=1 and T=7 sparse signed
fixtures. Unit latents and a single signed 127 coefficient per Q column give
the closed form `BF16(sign * 127 * per_channel_scale)` before RoPE. Distinct
head/channel scales yield RMS about 0.58–2.36, and all positions have
nonidentity RoPE. `golden_mla_semantic` uses this closed form plus complex
multiplication; it does not call the ordinary candidate projection golden or
its quantization, RMS or GEMM helpers. Both goldens agree **exactly** with the
executed pinned-source oracle on these fixtures. Default device checks require
exact Q/KV/qr equality and the ordinary scale tolerance. Existing random and
consumer thresholds are unchanged.

## Executed acceptance

All commands ran from repository root after `source activate-pypto.sh`.
All NPU work used `task-submit --device auto`; all artifacts/builds stayed here.

| Queue task | Execution and result |
| --- | --- |
| `task_20261106_111707_16915327213` | CPU semantic baseline T1/T7 and lint, exit0; `host_semantics.json`, `host_queue.log` |
| `task_20261106_111755_169402812215` | Device0, epochs1793935078–1793935411, all sixteen executable exits0: two mutation controls, three MLA cases, eighteen phase cases across all twelve entries, six indexer cases; `results.tsv`, per-entry logs |
| `task_20261106_111933_17051595550` | CPU rev3 imports/pins and model-directory ruff, exit0; `pins_queue.log` |
| `task_20261106_112048_171577817024` | CPU complete decode/prefill compile, epochs1793935252–1793935306, both exit0; `compile.tsv`, `full_compile/` |
| `task_20261106_112606_174756710941` | CPU artifact/source/provenance collector, exit0; `collect_queue.log` |
| `task_20261106_112859_175465031455` | Final CPU oracle rerun into separate output, exact prior-oracle match, script syntax/lint and collector, exit0; `final_host_queue.log` |
| `task_20261106_112350_174001929367` | Device0, asymmetric random T17 seed2906 against the executed original projection/W8A8/rotary reference, all four outputs PASS with existing gates, exit0; `baseline_queue.log`, `baseline_device.json` |

The component run has **27 positive numerical cases**, plus the separate
independent random device case gives **28 positive cases**. Their phase
agreement establishes propagation of F01; it does not establish complete
original attention semantics while F02–F04 remain open. The same compiled
dynamic MLA artifact is reused for default random T32 and analytic T1/T7;
there are three MLA build directories total: original mutant, FP32 mutant,
and the repaired artifact. `generated_evidence.json` shows repaired BF16
Q scratch, Wq_b output binding, direct BF16 RoPE consumption and only the
query-latent/KV RMSNorm tasks. Builds show lowering; numerical results supply
runtime proof.

Both meaningful device mutations fail on **Q only**; KV/qr/qr_scale pass:

- Frozen original head-normalized implementation: **57,344/57,344** Q
  elements fail exact equality, max absolute error3.078125.
- Remove only the head norm but retain FP32 until RoPE: **2,255/57,344**
  Q elements fail, max absolute error0.03125. This independently protects
  the cast boundary; the ordinary ratio gate can hide such BF16 differences.

Control runners exit0 only after confirming a numerical Q rejection. They
propagate compile/runtime exceptions; there was no unexpected numerical,
compile or runtime failure in this unit. Original sources, the FP32 mutation,
raw failure logs, reference data and comparator diagnostics are retained.
The original mutation includes both pre-repair mistakes; the second mutation
isolates the cast omission after removing normalization.

Exact reruns (do not run original/FP32 files as acceptance goldens):

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/attention_f01_1793934665/check_semantics.py host'
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/attention_f01_1793934665/run_components.sh'
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/attention_f01_1793934665/run_compile.sh'
```

For an independent fresh observer rerun, pass a new absolute output directory
as the first argument to `run_components.sh`; its final evidence-only plumbing
supports this without overwriting the original logs. For the reference helper,
set both `F01_EVIDENCE_DIR` and `PYPTO_PROG_BUILD_DIR` to fresh locations.
`check_semantics.py baseline -d "$TASK_DEVICE"` executes
the original-source random device check. Model default itself runs all three
permanent projection cases through `mla_proj_test -> mla_proj`.

## Limits and exact recovery

This completes the first coherent **F01** implementation/device milestone.
It does not close F02/F03/F04. Next unit starts with inverse query-position
RoPE of the attended value **before** Wo_a/Wo_b, shared across all twelve
phase paths and their goldens, including `layer_reference._partial_rows`.
Use the pinned upstream `_apply_rotary(..., inverse=True)` at
`dsa_v41.py:711–716`, nonidentity query RoPE and asymmetric Wo weights;
retain a missing-inverse negative control. Preserve F01's BF16-before-RoPE
boundary and analytic/default acceptance.

Then retain unrotated compressed latent for index-key projection while
publishing a separately source-rotated long-attention KV (F03, upstream
`:544–564`), and derive C2 rotary coordinates from emitting **absolute
position-1**, including odd-start continuation, rather than compressed row
address (F04, upstream `:1340–1349`). All selected modes, physical request
mappings, active tokens, thresholds, TP4/DP4/EP16 and W8A8 policy remain fixed.

Both complete-model **device** gates and ordinary sixteen-device layer suites
need fresh validation after semantic repairs. Standalone TP, composed decode
pool publication/state ownership, remaining F07 2048-block capacity, dynamic
lifecycle, CI ownership and generation remain OPEN/HELD. Component success
cannot close them. No original checkpoint weights were used in this bounded
unit; low-amplitude/half-tie quantization parity with torch_npu is not claimed.
Existing issue ledger was read; this own-code repair introduced no upstream
failure, toolchain bump, external filing or issue-status change.

`before.json`, `after.json`, `changes.diff`, `unit_summary.json` and
`generated_evidence.json` record source scope and builds. Final MLA source SHA256 is
`c8acc26cf76936995e049d51cc163f9aebf4acc332f6901d13d0302a6458592b`.
Model manifest SHA256
is `9352242dc8b16f94343ed6cc9c0e05e16d69ad3a92ca7c0ee65d408898f16881`.
The native goal was observed null at entry and closeout; none created/completed.
No checkpoint/PyPTO write, commit or subagent. Independent observer acceptance
of this new F01 milestone is not claimed. The unit parks after final evidence
collection and state-carrier updates; final task/lock status is in TASKS.
