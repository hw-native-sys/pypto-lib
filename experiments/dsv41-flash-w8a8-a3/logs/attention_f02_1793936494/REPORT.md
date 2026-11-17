# F02 bounded repair: inverse attention-output RoPE

The shared output projection now applies the inverse query-position rotary
transform to BF16 attended heads before grouped Wo_a, narrows its intermediate
to BF16, then applies Wo_b. The existing rotary helper consumes an explicitly
negated sine table and writes a separate BF16 restored-head scratch buffer.
Its forward implementation and all accepted MLA/indexer sources are unchanged.
All eleven direct prefill/decode phase call sites pass their query cosine/sine
tables. Decode C1 reuse delegates to the shared decode reuse implementation.
Both layer-reference `_partial_rows` call sites supply the same query tables.
Public phase/layer/forward ABIs, quantization, paged addressing and topology
are preserved. F03 compressed-value basis and F04 C2 source coordinates remain
open and are outside this unit.

## Independent semantic grounding

Official vllm-ascend rc2 revision
`8fde6d4e335fe300e2b949aab0e31659cce0811d` remains the original A3 authority.
Worker `independent_oracle/REPORT.md` and `host_results.json` record four source
hashes and verify their exact bytes against the preserved official archive,
SHA256 `48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`.

- `dsa_v41.py:711–716` inverse-rotates the attended values at query coordinates
  before output projection; `_apply_rotary:291` passes direction and head range.
- `device_op.py:76–90` supports inverse direction through negate_sin or -sin.
  The original rotary CPU golden uses FP32 adjacent-pair arithmetic and preserves
  the nonrotary prefix. Copying into the BF16 inplace destination narrows the
  rotated values before projection.
- `dsa_v1.py:_wo_a_bmm` and `_forward_o_proj` use the non-FP8 BF16 grouped BMM
  path then Wo_b. The worker AST-executes these unmodified source methods,
  the original device adapter and original rotary CPU golden, with explicit
  CPU stand-ins for NPU primitives.

The separate sparse oracle uses complex multiplication and direct three-term
Wo_a/two-term Wo_b equations. It imports no candidate helper. Asymmetric signed
weights, rotary/nonrotary channels, distinct groups/heads and varying nonidentity
angles make rotation direction observable. Original-source and closed-form
outputs match exactly at T1/T7/T17. Main's ordinary and permanent semantic
goldens also match that original reference exactly at those shapes.

The permanent `o_proj.py` default now tests random T16 and exact sparse T1/T7/T17
through the real shared implementation, reusing a single dynamic compiled
artifact. This semantic golden does not call the normal rotary/projection
golden or compute helpers. Existing gates remain unchanged; exact sparse checks
add coverage. `check_layer_reference.py` independently derives the attention
sum for one visible row and zero logits, then compares `_partial_rows` with
the original-source output oracle exactly at T1/T7.

These source-backed CPU stand-ins do not execute original torch_npu primitives
on hardware or the original checkpoint/full model. Exact primitive reduction,
low-amplitude/tie quantization parity, TP/DP collectives and real-checkpoint
generation parity are not claimed. F01's explicit quantization limitation persists.

## Execution and failures

All CPU/compile/lint checks use `task-submit --no-device`; NPU checks use
`task-submit --device auto` with the queue's TASK_DEVICE. Environment activation
and pinned PTOAS are explicit in runners. Rev3 pins verified:
PyPTO38a8aa6, simpler31f9e72, PTOAS0.72, ISA f83d4b7. No environment changes.

| Queue task | Scope |
| --- | --- |
| task_20261106_114433_191733421126 | Snapshot all45 model sources, exit0 |
| task_20261106_114552_1950517298 | Worker CPU original-source/analytic oracle, exit0 |
| task_20261106_114605_196168324371 | Caller adaptation, model lint/imports, exit0 |
| task_20261106_114654_198763920693 | Initial random output-projection smoke on device0, exit0 |
| task_20261106_114718_199009922828 | Permanent semantic default integration/lint, exit0 |
| task_20261106_114941_205461625637 | Initial integration sweep, stopped by Main, exit130; failures preserved |
| task_20261106_114941_205461820046 | Initial full-forward compile attempt, exit1; failures preserved |
| task_20261106_115125_211317811282 | Second integration sweep, exit1; failures preserved |
| task_20261106_115125_211317714201 | Second full-forward compile attempt, exit1; failures preserved |
| task_20261106_115421_216587328948 | Corrected component/control sweep, device0; see device_v3/results.tsv |
| task_20261106_115421_216584112484 | Both complete forwards compile, exit0; compile_v3/compile.tsv |
| task_20261106_115605_219038817434 | Independent layer-reference oracle, model lint/runner syntax, exit0 |
| task_20261106_120152_230130317548 | Final source/evidence collector and model lint, exit0 |

Corrected sweep final result: all19 executable exits0,33 positive numerical
cases and2 expected numerical mutant rejections, on device0 at
epochs1793937265–1793937688. The33 cases comprise4 output-projection defaults,
3 MLA defaults,18 phase cases,6 indexer cases, and2 original-source T17 cases.
Full-forward compile epochs1793937266–1793937322, both exits0. The permanent
operator's four runs use one dynamic build artifact. No model source changed
after this corrected sweep began.

Final45-source manifest `after.json` SHA256
`d36bb70d6d4b285785fb92c7c435f8fd99651b895ed91e3a32a675acb016cf7f`.
Exactly13 sources change: o_proj.py,11 direct phase files and layer_reference.py;
the other32 are byte-identical. Final o_proj.py SHA256
`11c2c872c91f485bc91f68d3ea4ae08672d3395539e2b390f147efd650a29668`.
Protected MLA remains
`c8acc26cf76936995e049d51cc163f9aebf4acc332f6901d13d0302a6458592b`,
indexer remains
`8f998dc7ae8773e67acda854af515f9ae7625374ba101b71152e243b0585df43`.

Initial failures were Main integration mistakes, not successful numerical
controls: PyPTO child constexpr calls do not read signature defaults; the new
sparse harness specs initially used dictionary order instead of entry argument
order; and config attribute expressions were initially placed inside a scratch
shape where DSL parsing requires module constants. These were corrected without
modifying protected MLA callers or relaxing gates. Final `rope.py` is byte
identical to F01. No upstream bug attribution or workaround is claimed.
The first multiline snapshot submission (task_20261106_114407_190449211752)
returned exit0 with a heredoc warning and did not create a snapshot; its
replacement script task above did. Final acceptance never relies on that attempt.

Corrected device controls numerically reject the missing-inverse mutant at
81523/87040 output lanes (maxabs2.53125) and the wrong-sign mutant at
79691/87040 lanes (maxabs2.5546875), at unchanged atol/rtol0.02 and <=0.5%
outliers. The runner requires an output mismatch, so compile/runtime exceptions
cannot count as a successful control. The original-source sparse and random
T17 checks pass, as do permanent operator, MLA and phase/indexer defaults.
Final exact case counts and source hashes are in `unit_summary.json`.

Generated orchestration in `generated_evidence.json`/`generated_excerpts.txt`
shows FP32 negative-sine scratch produced before BF16 rotary, BF16 restored
heads bound into Wo_a, and BF16 Wo_a intermediate bound into Wo_b. Both
complete-forward compile artifacts contain the shared inverse-output path.
This is lowering evidence; device results establish the exercised numerical
boundary. Complete-model runtime publication/state correctness is not proven.

## Reproduce without overwriting evidence

From repository root, choose a fresh absolute output directory:

```bash
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/attention_f02_1793936494/run_components.sh /absolute/fresh/device'
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/attention_f02_1793936494/run_compile.sh /absolute/fresh/compile'
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && F02_EVIDENCE_DIR=/absolute/fresh/host python experiments/dsv41-flash-w8a8-a3/logs/attention_f02_1793936494/check_semantics.py host'
```

For individual device checks set F02_EVIDENCE_DIR and PYPTO_PROG_BUILD_DIR to
fresh paths and invoke `check_semantics.py missing|forward|baseline|sparse -d
"$TASK_DEVICE"` inside an allocated task. `baseline` is random T17 seed2907;
`sparse` is independently referenced asymmetric T17. Helpers resolve the
preserved reference next to this unit; do not move only one script away.

## Recovery and limits

This is Main implementation/self-validation evidence, not independent acceptance.
Observer assigns the sibling reviewer/validator to rerun acceptance on the final
frozen source manifest and records the decision. Main-owned oracle worker
`/root/main_native/f02_oracle` is parked; its unique work record was imported
once into TIME_LOG and parallel time was not summed as elapsed time.

Next authorized unit after acceptance is F03: preserve unrotated compressed
latent for index-K projection and publish separately source-rotated attention
KV, grounded in original `dsa_v41.py:544–564`. F04 then derives C2 source
rotary rows from emitting absolute position-1, including odd-start continuation.
Do not continue either under this F02 unit. Preserve physical mappings, dynamic
axes, active token counts, W8A8/topology and every gate.

F03/F04, Engram F05/F06, remaining2048-block capacity, dynamic routing/lifecycle,
generation F11–F13 and conditional offset incidence remain OPEN. Both
complete-model device gates, ordinary16-device layer suites, standalone TP,
CI ownership and composed decode pool/state diagnosis remain OPEN/HELD.
Generation remains HELD; historical full-model PASSes cannot accept this tree.
No sixteen-device layer/full-model run or original checkpoint weight use is
claimed by this bounded unit. No commits, external filing, checkpoint/PyPTO
writes or simulator acceptance. Existing issue ledger inspected; this own-code
fix changes no toolchain/known issue status and introduces no upstream defect.

STOP: model manifest frozen. Main's worker and own queue jobs have ended;
device0 allocation released. No unit-owned background implementation writer
remains. Recovery is this report, TASKS checkpoint, after.json, unit_summary.json,
per-entry logs and append-only TIME_LOG. Await independent sibling validation
and observer's next bounded-unit assignment before any further model writes.
