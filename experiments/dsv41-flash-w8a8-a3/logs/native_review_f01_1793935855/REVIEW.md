# Independent native review: bounded F01 common MLA

Disposition: no blocking F01 source defect found in the final manifest below. The repair removes the unintended head normalization and restores BF16 narrowing before query RoPE, consistently in the shared kernel and owning golden. Original extra normalization and the normalization-removed FP32 mutant are independently distinguished. This is a source/CPU review plus inspection of retained device evidence; it does not authorize acceptance or close downstream gates.

## Final snapshot actually reviewed

Final CLI REPORT.md, after.json, artifacts.json and unit_summary.json were present, rather than pending. Observer subsequently confirmed CLI STOP and PID1673534 exit, and no implementing writer. Independently verified all 45 current model file hashes against after.json and all 42 entries of artifacts.json. Only mla_proj.py differs from the before manifest; the other 44 files, including indexer and padding comparator, are unchanged.

- MLA SHA256: `c8acc26cf76936995e049d51cc163f9aebf4acc332f6901d13d0302a6458592b`
- after.json SHA256: `9352242dc8b16f94343ed6cc9c0e05e16d69ad3a92ca7c0ee65d408898f16881`
- final CLI REPORT.md SHA256: `a68ad56b5258aa6ec21e06d284e6dde78bde26f14c7841a8ba0e8c2347dd44d4`
- check_semantics.py SHA256: `bf42a4d94a396238665c560d6421245774f2f89e59ab725805569151d2e42ad4`
- FP32 mutant SHA256: `fb0ed3cd3ac642d4f8880c1cc23b5b177dceb521f66662fa03084756de27cab0`

Reference is pinned vllm-ascend v0.30.0rc2 revision `8fde6d4e335fe300e2b949aab0e31659cce0811d`, from the prior independent review archive. Verified archive SHA256 `48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c` and independently checked byte equality of all three executed upstream sources to tar members. This review inherits the archive's previously established official origin; it did not redownload it. Exact upstream file hashes are in cpu_results.json.

## Semantic review and negative controls

Primary source `upstream_rc2/vllm_ascend/attention/dsa_v41.py:320–331`, `_project_q`/`_project_kv`, computes query-latent weighted RMSNorm, Wq_b, then partial rotary directly. No post-Wq_b head norm occurs. KV keeps its own weighted norm. `w8a8_dynamic.py:104–135`, `AscendW8A8DynamicLinearMethod.apply`, passes BF16 activation dtype as quant-matmul output_dtype in the ordinary path, and preserves the recorded output dtype for prepared input. Therefore Wq_b produces BF16 before rotary for the selected stream. The source rotary CPU golden uses FP32 arithmetic on those already-narrowed values and preserves nonrotary lanes. An unused declared norm member cannot alter this execution equation.

Reviewed mla_proj.py:65–80, `_golden_mla_proj`, `golden_mla_semantic` and both fixtures. Kernel now binds a BF16 Wq_b result directly to BF16-source rotary. Owning golden's gemm narrows before reshape/RoPE. Query latent quantization, KV path and signatures remain intact. All twelve phase entries share this kernel (decode C1A reuse delegates through decode C2A reuse); phase/layer reference imports share the owning golden. Their parity is propagation evidence, not independent original-attention correctness.

The oracle AST-executes the pinned original projection, W8A8 apply method and rotary CPU golden. NPU quantization/matmul and RMSNorm are CPU mathematical substitutions; model NZ unpacking and configuration dimensions are shared plumbing. It calls no candidate quantization/GEMM/RMS/RoPE golden helpers for the reference. T1/T7 signed sparse fixtures have nonidentity rotary phases, alternating token signs, signed columns and distinct per-head/channel magnitudes. Unit BF16 latents give an independently checkable `BF16(sign * coefficient * 127 * scale)` query before rotary. Exact host oracle, closed-form complex-multiply golden and ordinary golden agree for both fixtures, verified again in this review.

Independent review_cpu.py implements direct adjacent-pair arithmetic and separately constructs normalized/FP32 equations. T7 results:

| Equation | Q mismatches | Nonrotary | Rotary | Max absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Original extra norm + FP32 rotary | 57,344 | 50,176 | 7,168 | 3.078125 |
| Norm removed, FP32 rotary retained | 2,255 | 0 | 2,255 | 0.03125 |
| Norm retained, BF16 boundary restored | 57,344 | 50,176 | 7,168 | not separately recorded |

The last construction isolates normalization with the correct cast boundary. All 50,176 nonrotary original mismatches independently identify the extra normalization: omitting a pre-RoPE cast cannot explain nonrotary errors, because the final output cast gives the same value there. FP32-only failures isolate the rotary boundary after norm removal. First two counts exactly reproduce retained device logs. Original/FP32 mutation source diff removes only head normalization; its unused imports/golden do not affect the runner, which explicitly supplies the independent semantic fixture golden. Raw logs confirm numerical Q rejection with KV/qr/qr_scale passing, rather than compilation or runtime failure being counted as a negative-control pass.

## Prominent limitation: exploratory random equality fails

This review's first extra CPU check required bit-exact ordinary-golden/original-substitution agreement on random T17 seed2906 and failed. Preserve cpu_queue.log (task_20261106_113238_17997284811, exit1). It was an exploratory assertion stronger than the existing random acceptance gates. The follow-up records the disagreement rather than discarding it:

- Q: 28,568/139,264 exact mismatches, max0.0625; 146/139,264 (0.10484%) outside existing atol0.02/rtol1/64, below the existing1% gate.
- KV: 649/8,704 exact mismatches, max0.015625; zero outside that tolerance.
- qr: 141/21,760 exact mismatches (0.648%), each at most1, below existing1% allowance. qr_scale differs at7/17 values by at most3.73e-9.
- Even input dynamic quant differs at25 INT8 values. Oracle uses divide-by-amax then multiply127; model uses multiply-by-reciprocal. Scale arithmetic differs slightly. Half ties and downstream BF16 boundaries can amplify this difference.

These results establish neither exact torch_npu quantization parity nor a new F01 regression: the original NPU primitive is substituted in this oracle, and the bounded repair leaves quantization untouched. Actual original torch_npu primitive parity, zero/near-zero behavior and tie handling remain unproven. Main's report explicitly excludes low-amplitude/half-tie parity; preserve this limitation during subsequent units. Random candidate/golden agreement must not replace independent original primitive/checkpoint validation.

## Generated and device evidence boundaries

Inspected repaired standalone orchestration: q_bf16 scratch uses DataType::BFLOAT16 at line69; Wq_b binds that output at184; direct rope_tail_bf16 input at204; exactly query-latent and KV RMSNorm tasks remain. Inspected kernels/aiv/w8a8_gemm_0_aiv.cpp:993–1001: FP32 scaled query narrows with CAST_RINT to bfloat16_t. Inspected rope_tail_bf16.cpp:169/380/393: BF16 query load precedes FP32 rotary arithmetic; nonrotary values copy through. This is lowering evidence, not runtime proof. Generated files reviewed are hashed in reviewed_files.sha256.

Retained results.tsv and raw logs support Main's27 positive component cases plus1 independent-source randomT17 case, two rejected device mutations and two full-forward CPU compiles. Those device runs are Main's evidence, not review-agent runs. Observer separately reports fresh personal NPU acceptance task_20261106_113314_180195711346 exit0 with both mutants, random original-source baseline and three permanent MLA cases; observer owns that decision and evidence. This review submitted no NPU work.

## Checks, ledger and parked resumption

Only CPU jobs were submitted, using task-submit --no-device after source activate-pypto.sh. Exact command is review_cpu.py under this directory; cpu_queue.log and cpu_queue_rerun.log preserve queue task IDs and exits. Rerun task_20261106_113402_18059792208 exit0 re-executes exact analytic oracle checks, archive/source/artifact hashes, independently separated mutation equations and quantified random discrepancies. model_manifest.json and cpu_results.json are durable review outputs. No model, golden, threshold, Main evidence, TASKS or checkpoint/PyPTO files edited; no worker, commit or external filing.

Parked at bounded F01 review milestone. Remaining F02–F04 are explicitly open; layer/full-model device, full dynamic lifecycle, TP/DP/EP interactions and real-checkpoint generation are not established by this review. Observer may use this review alongside its own acceptance, then assign F02 separately. On resume verify current manifest and live ownership first; preserve both BF16 boundary and exact signed T1/T7 defaults, and retain the quantization parity limitation. Initial reading before measured epoch1793935855 is unmeasured; TIME_LOG contains only measured review-agent intervals.
