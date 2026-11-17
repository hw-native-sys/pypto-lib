# Independent frozen-cohort acceptance

All assigned independent checks passed, with expected numerical rejections for the negative controls. No blocking bounded F03/F08/conditional-F14 source or numerical defect found. Recommend bounded acceptance to observer: F03 basis/publication and F08 rank-local legal-tie routing; F14 only for representable synthetic offset folding. Observer records the decision; this review closes no gate.

Detailed semantic/source findings and limitations are in SOURCE_REVIEW.md. All tests here were personally executed by observer-owned reviewer /root/reviewer_native, independently of Main and its two workers. Main/worker prior self-tests are not counted as these runs.

## Actual frozen snapshot

Verified supplied final report SHA256509d2e6a8a009de225863de1e1717668476ddf4c76803954974d54b3d4c11dda, after.json SHA2560bba0d6b19f91da52bf5c23e466cbf44d7edbc326e5dadeb25f3ecd43d7ad6f2, artifacts.json SHA256d50c5d13a65e0f3f5107fa5cc6647498e1a6709a9a169256a573af6ce566b6b7. All45 current sources and128 Main evidence files plus both worker manifests match. Ten independently saved source manifests agree before/after every batch and at final collection. The nine-file scope is exactly the four source phases, rope.py, paged_cache.py, layer_reference.py, moe_gate.py and weight_loader.py. Protected MLA/indexer/o_proj bytes and both accepted rotary-kernel ASTs are unchanged. frozen_source/ preserves all45 reviewed files.

Final F08 source040019c27390f3bc5355af1b1964c19dc3ec347a6e719fe795adedc047b4f872; F14 source46d42683c2450ebb49947df2bd823dc3659d66cf5ba4c5ce12d0f92faf51aa9e. Per-file hashes are source_final.json. Reference remains original rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d; original archive/source bytes checked again in personally executed CPU helpers.

## Personally executed results

Exact submissions are commands.sh; run.sh records wrapper epochs/allocation and verifies frozen hashes. All CPU/compile tasks use --no-device; both NPU tasks use --device auto and TASK_DEVICE. Environment activation and pinned PTOAS are explicit. Host logs retain PyPTO38a8aa6/simpler31f9e72/PTOAS0.72/ISA f83d4b7. No simulator or16-device job was used.

| Queue task | Independent result |
| --- | --- |
| task_20261106_155547_348391322037 | Initial45-source/128-evidence/both-worker/protected-hash/rotary-AST verification, exit0 |
| task_20261106_155650_34865824977 | F03 auto device0, epochs1793951815–1793952442:4 frozen-before numerical cache rejections,4 repaired positives,4 wrong-index-basis numerical rejections; all12 entry exits0; actual queue task exit0 |
| task_20261106_155650_348658328755 | Auto device1, epochs1793951815–1793952250:17-executable/37-positive downstream sweep, plus3 F08 invalid-ID before rejections and7 separate F08 positives (4 defaults,3 repeated tie cases); exit0 |
| task_20261106_155650_34865856081 | CPU original-source F03/latent equations, original-source F08 comparator checks, staged/cache integration and synthetic F14 before/after checks; epochs1793951816–1793951847, exit0 |
| task_20261106_155650_348658411491 | Both complete forwards compile; wrapper epochs1793951815–1793951877, decode1818–1848/prefill1848–1876, exit0 |
| task_20261106_160005_351991512686 | Additional reviewer cache-gate controls at widths128/512, including duplicate/sentinel/empty publications; exit0 |
| task_20261106_160750_357089925872 | Final source/evidence verification, numerical-result and generated-binding collector; exit0 |

The37 downstream positives comprise3 MLA,4 o_proj,4 router,2 compressor,6 indexer and18 cases across all12 phases. F03 adds4 source-backed positives. F08 separate replay adds7 positive executions with repeated fixtures; do not count these as7 distinct test designs. F03's8 negative controls and F08's3 negative controls are separate from positives.

Raw logs and results.json verify actual numerical failures: every F03 frozen-before control contains compressed_cache FAIL; each wrong-index-basis control contains index_cache FAIL; each F08 frozen-before control rejects invalid or duplicate expert IDs. No compilation/runtime exception counts as a successful numerical control. Per-output verdicts, error/lane counts and build paths are in results.json and raw logs. Device helper result.passed=False is the correct negative-control outcome, rather than a failed acceptance runner.

The F03 foreground task-submit client returned exit1 after its default600-second WAIT timeout, while the job continued. Its actual627-second job completed exit0; f03_execution.txt, all12 result rows, final hash check and f03_status.txt confirm this. Timeout text is preserved in f03_queue.log. No duplicate rerun was submitted and no numerical failure was hidden.

F03 original flow/complex rotation agrees exactly at C1/C2 T1/T7/T17 and captured index input remains raw; separate compressor equations match all four source fixtures. The strengthened cache comparator retains the full-cache gate plus same-band published rows; dilution failure rejects. Reviewer-specific empty/all-skipped tests show the whole-cache gate remains enforced.

F08 original-source CPU checks accept legal alternative tie orders and reject invalid/duplicate IDs, missed biased winners, wrong weights and biased normalization. Staged tests cover equal biased scores with different unbiased scores/legal subsets. Existing weight band remains unchanged.

F14 synthetic before loses18432 NZ bytes over9 nonzero matrices, while9 zero matrices are exact. Repaired18 zero/nonzero payload samples,72 shard matrices/72 scales and2INT8 endpoints match independently; fractional and both overflow folds reject; owning validator reports zero errors. No real checkpoint was read. Current-checkpoint nonzero-offset incidence is unestablished.

## Generated evidence and disposition limits

Fresh generated.json hashes/excerpts cover all four source components and both complete-forward orchestration files. Directly inspected C2 component binding: raw latent feeds rotary into distinct BF16 attention_latent; snapshot consumes attention_latent and publishes its result to compressed slots with explicit snapshot dependency; index projection separately consumes the raw latent. Both complete-forward compiles retain that split. Generated code establishes lowering/bindings, not complete-model runtime publication or distributed state ownership.

Original torch_npu compressor/rotary/reduction/routing arithmetic remains modeled on CPU. Exact primitive and low-amplitude/tie quantization parity are not established; F01's exploratory random bit-equality failure remains a limitation. F04 absolute C2 coordinates is NOT repaired or accepted by this supplied-table basis test. F09/F10 routing/dynamic lifecycle, Engram, remaining capacity, full-model/16-device layer/standalone TP/CI/pool-state gates and real-checkpoint generation remain OPEN/HELD. No current-checkpoint corruption claim follows from conditional F14.

All own jobs completed; allocations released. No model/golden/threshold/Main evidence/TASKS/checkpoint/PyPTO writes, workers, commits or external filing. Measured review-agent ledger rows use the shared flock; parallel job times are not summed as elapsed time. Reviewer PARKED. Observer may record separate bounded dispositions and assign the next frozen scope; on resume verify current ownership, protocol/directives and source manifest before execution.
