# Independent F02 source review and executed acceptance

Disposition: all assigned independent checks passed; no blocking bounded-F02 source/CPU/device defect found. The evidence supports rank-local inverse attention-output RoPE before grouped output projection. Observer records the acceptance decision. This report does not close any gate or establish F03/F04/full-model semantics.

## Frozen source and evidence actually reviewed

Main/worker were PARKED with ended self-test jobs when observer assigned this execution. The final REPORT/manifest were available, not pending. Verified supplied hashes and all45 current model sources plus all156 Main evidence entries before testing, before/after host/device/compile batches and at final collection. Eight independently saved source manifests agree. Exactly13 authorized sources changed from F01: shared o_proj, eleven direct phase files and layer_reference. Shared rope, MLA and indexer remain byte-identical. A reviewer-owned copy of all45 model sources is frozen_source/.

| Artifact | SHA256 |
| --- | --- |
| Main final REPORT.md | a55280585e1bbc917725855c3ac288e418f879b3b0bb61a4ad9b507c26629656 |
| Main after.json | d36bb70d6d4b285785fb92c7c435f8fd99651b895ed91e3a32a675acb016cf7f |
| Main artifacts.json | 0ff6deda1fe501da84fc73be881da65811e05508142b08e1e0f9c5cc1cf81e8f |
| o_proj.py | 11c2c872c91f485bc91f68d3ea4ae08672d3395539e2b390f147efd650a29668 |
| mla_proj.py | c8acc26cf76936995e049d51cc163f9aebf4acc332f6901d13d0302a6458592b |
| indexer.py | 8f998dc7ae8773e67acda854af515f9ae7625374ba101b71152e243b0585df43 |
| rope.py | d21e15e8bb37f5f49a1dd4257a276b2c9de1d6b1e51f7a521e38c903c0b85e3d |

All tests below were personally submitted/executed by observer-owned /root/reviewer_native in this new output tree. Main and its oracle worker's prior test results are not counted as these independent runs. No model/golden/threshold/Main-evidence/TASKS/checkpoint/PyPTO files were edited. No worker, commit or external filing.

## Independent source and CPU findings

Pinned authority: vllm-ascend v0.30.0rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d. Reverified archive SHA25648c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c and exact tar-member equality for all four executed reference files. Individual source hashes are in host/host_results.json. Archive origin is inherited from the prior review; no redownload was necessary.

Original dsa_v41.py:711–716 applies inverse query-position rotary to BF16 attended heads before _project_output. device_op.py:76–90 maps inverse to negate_sin or explicitly negated sine. Original rotary CPU golden preserves the nonrotary prefix and computes adjacent pairs in FP32 before destination BF16 narrowing. dsa_v1.py:_wo_a_bmm/_forward_o_proj gives grouped BF16 Wo_a followed by Wo_b for the selected non-FP8/rank-local path. Actual Wo_a/Wo_b checkpoint metadata is BF16. Optional output-projection collective branches are outside this rank-local proof.

Reviewed o_proj.py:38–99 and both owning goldens: separate FP32 negative-sine scratch, separate BF16 restored heads, BF16 Wo_a intermediate and BF16 final output implement the expected order. Original attended input is read into distinct scratch; no source-level in-place mutation or bypass into Wo_a. Eleven direct phase callers/goldens supply query cos/sin, delegated decode C1 reuse reaches the same corrected implementation, and both layer-reference call sites pass rank-local query tables. Public phase function signatures and all pre-existing phase comparator ASTs are independently unchanged. Existing ordinary o_proj gates are retained; exact semantic defaults add coverage.

Personally reran preserved-original AST adapter/rotary/grouped-projection methods with explicitly documented CPU primitive stand-ins. Closed-form sparse oracle, original-source execution, ordinary candidate golden and permanent semantic golden agree exactly at T1/T7/T17. Fixtures use signed rotary/nonrotary lanes, distinct groups/heads, asymmetric three-term Wo_a/two-term Wo_b and varying nonidentity phases. The permanent model semantic oracle originates from Main's worker, so independence comes from original-source grounding plus separate reviewer checks, not its authorship.

Reviewer-owned review_cpu.py additionally derives inverse real/imag equations and dense per-group contractions without candidate compute/golden helpers: exact original-source agreement and unchanged nonrotary prefix at T1/T7/T17. A separate omission of the pre-Wo_a BF16 narrowing changes1,604/12,017/29,577 final values respectively (maxabs0.0078125), demonstrating why exact fixtures protect this boundary better than ordinary loose random gates. Independently derive one-visible-value attention with zero Q/sink as value/2, then test actual layer_reference._partial_rows against original output oracle: exact at T1/T7/T17; missing inverse is distinguishable. The supplied Main layer-reference script writes to its own directory; it was not executed. Reviewer adapter preserved equivalent and extended checks in fresh host output.

## Independent execution matrix

Exact shell submissions and full runners are commands.sh/run_*.sh. All run from repository root after source activate-pypto.sh, explicitly using pinned PTOAS_ROOT/bin. Host recorded PyPTO38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e, simpler31f9e72ee6832cd198b77df18c1c305e7fd83676, PTOAS0.72, ISA f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7.

| Reviewer queue task | Result |
| --- | --- |
| task_20261106_142605_295199121033 | --no-device initial frozen45-source/156-evidence verification; exit0 |
| task_20261106_142738_295992014521 | --no-device original/analytic/candidate semantic checks and independent layer checks; exit0 |
| task_20261106_142738_295992130727 | --no-device both full forwards compile; exit0; decode epochs1793946466–1793946495, prefill1793946495–1793946522 |
| task_20261106_142738_295992226918 | --device auto allocated0; wrapper epochs1793946463–1793946880; all19 executable exits0,33 positive numerical cases plus2 expected numerical rejections |
| task_20261106_143552_301943820852 | --no-device final unchanged hashes, per-output result/generation collector; exit0 |

The33 positives comprise original-source randomT17 and sparseT17 (2), permanent o_proj randomT16/exactT1/T7/T17 (4), accepted MLA defaults (3), all12 phase entries (18), indexer defaults (6). Every output verdict is retained in results.json and raw device/*.log; device/results.tsv gives actual per-entry start/end/device/exit. No unexpected failure occurred in this independent run. Main's earlier integration failures remain historical Main evidence and were not counted as successful controls.

| Device mutation at unchanged output gate | Numerical rejection | Maximum absolute difference |
| --- | --- | --- |
| Missing inverse | 81,523/87,040 lanes (93.6615%) | 2.53125 |
| Forward instead of inverse | 79,691/87,040 lanes (91.5568%) | about2.55469 (CPU oracle2.5546875) |

Both raw logs contain output FAIL with ratio_allclose atol0.02/rtol0.02/max_error_ratio0.005 and explicit mismatch diagnostics; no compile/runtime exception was treated as success. Mutant source files, error JSON and build/data artifacts remain under device/. Counts match the original-source CPU semantic controls. Positive original-source device checks use these same unchanged random gates; permanent sparse T1/T7/T17 require exact equality.

## Generated-source findings and practical limits

Fresh generated_review.json records hashes/excerpts for standalone and both complete forwards, plus relevant standalone kernels. The permanent o_proj four cases reuse one dynamic build `_jit_o_proj_test_20261106_142920_b1o4qeib`. Its orchestration produces inverse sine, binds negative sine and input attended into separate restored heads, binds restored BF16 heads to Wo_a/mid at105–107, and binds BF16 mid to Wo_b/output at136–138. Allocated restored/mid scratch uses BFLOAT16. Generated output_inverse_sine.cpp contains TNEG; rotary and output-projection generated kernels show BF16 pointers and CAST_RINT narrowing. Fresh full-forward orchestration includes the same inverse->restored->Wo_a->mid->Wo_b input/output chain across its lowered phase paths. This is lowering/dependency-binding evidence; phase numerical runs exercise the corrected boundary, while full-model runtime publication remains unproven.

Original torch_npu rotary/BMM/linear primitives are mathematical CPU stand-ins in the reference, not original hardware execution. Their exact reduction behavior, tie rounding and low-amplitude quantization parity are not established. Preserve F01's failed exploratory random bit-equality check and quantified modeled-quantization arithmetic gap; current success does not upgrade that to exact primitive parity. Fixed deterministic shapes/tails and rank-local defaults do not establish general serving lifecycle or all request/state geometries.

No16-device layer or full-model NPU job was submitted. F03/F04, Engram, remaining2048 capacity, dynamic routing/lifecycle, standalone TP, composed decode pool/state, CI and real-checkpoint generation remain OPEN/HELD. No checkpoint weights were used by these checks. Observer can decide bounded F02 from this independent evidence, then assign F03 separately. Reviewer PARKED after all own jobs completed; no live allocation or implementation writer. Resume only on observer assignment, rereading current protocol/TASKS and verifying the next frozen scope. Measured review-agent ledger rows are appended under the shared lock; parallel task durations are not summed as elapsed wall time.
