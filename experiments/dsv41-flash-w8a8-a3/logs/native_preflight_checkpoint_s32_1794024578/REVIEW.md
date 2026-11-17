# Independent artifact/CPU preflight — V2 checkpoint and exact S32 evaluator

Disposition: the four-callsite CPU exact-integer transformation has independent mathematical/source/small-numerical support. The frozen V2 checkpoint pipeline is NOT ready for native or large-IO authorization: it needs the explicit V3 fixes below. No V3 acceptance, full40 reference result, prefix17 acceptance, chain, fit, first text, or new hardware claim is made. All execution in this unit used task-submit --no-device. ROOT45 remains ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.

The sealed V2, evaluator and proof manifests were personally verified before copying, and source/evidence copies were kept separate. See verification.json and final_checks.json. Reviewer path adaptations only redirect source roots/output directories into owned copies; original sealed helpers are preserved. No model/golden/threshold/runtime/harness source was edited.

| Finding | Evidence and required next behavior |
| --- | --- |
| Offline mutable-state initialization is missing | Actual fresh copied prefill/decode imports expose `_MUTABLE_NAMES=[]`. Their exact original clone preludes then alias inputs. Small execution of those preludes reproduces mutation of initial state. Installing the exact source-owned contracts (prefix16, decode12), without materializing fixtures, isolates both passes. The future adapter must initialize these contracts and separately hash initial bytes before/after UNCONDITIONED and STAGED references; check distinct oracle clone storages. Existing normal fixture-builder native results are unaffected. |
| Pre-materialization geometry check absent | RealProvider.build_prefix immediately calls build_fwd_specs(...compile_only=False). Literal recipe expert_layers=1 does not validate actual imported compile-time EXPERT_LAYERS. Own META test confirms the intended one-expert geometry, but V2 does not enforce it before materialization. Fail closed on expert count, N128/pages2, all shapes/dtypes/argument coverage and byte totals before the first fixture call. |
| Budget receipt lacks semantics | Own guard-only test accepted an arbitrary hashed JSON object, required host/disk bytes1, and no SHM requirement. No fixture/provider was created in this control. Future receipt must bind schema/stage/source/helper/evaluator/static plan, actual HOST/DISK/SHM observations, derived required component budgets and explicit margins; reject missing/unknown fields. |
| Fixed-prefix retention misses later publications | V2 hashes only original128 WINDOW/C1 and64 C2 rows. Own tiny tests of the EXACT frozen decode published-cache comparator demonstrate that poisoned historical WINDOW128 or C2row64 can pass the full-cache1% ratio and current-step row checks at a later step. A growing publication set must preserve prior physical rows, excluding legitimate current writes, for every positive step. Own set-aware equality rejects these poisons; legitimate current writes remain allowed. |
| Evaluator receipt provenance is dropped | store.cpu_validate emits a generic receipt and discards evaluator receipt hash/explicit native_resume_authorized=False. The current RealProvider canonical-path check correctly rejects derived modules. A new explicit bridge must bind exact derived receipt/source/Torch hashes, mutable contract, both original gate workflows, initial immutability and source/static/ABI/owner/parent identities. Do not remove the path guard or treat CPU math alone as hardware authorization. |

The original published WINDOW and C2 comparator was also personally executed with correct129 oracle metadata and deliberately wrong currently published rows. Both reject numerically with the expected `published step rows` verdict. A Python/runtime exception is never credited as a control rejection. The first reviewer scaffold invocation omitted required comparator kwargs and raised TypeError; that failure is preserved, and the corrected fresh invocation passed. This tiny comparator exercise does not execute the full native stale-metadata control.

## Exact S32 assessment

The accepted domain is CPU rank2 INT8 or INT32 tensors whose values are within [-128,127], conforming matrix dimensions, and K<=131071. Each product has magnitude<=16384 and every partial sum is bounded by 16384*K<=2147467264<2^31. Integer products and all partial sums are exactly representable in FP64; conversion to S32 therefore equals the original nonoverflowing integer dot. INT32 operand casts, following FP32 scales, ordering and BF16 boundaries remain unchanged. The primitive rejects unsafe K or out-of-domain/rank/dtype/shape inputs; it does not silently approximate using FP32.

All four original consumers were checked by reversing ONLY the rewritten callee and removing its appended primitive import: their complete ASTs equal the frozen originals; the other41 modules remain byte-identical. Both composed-reference bodies, comparator/direct-pool/count workflows and thresholds are unchanged. No global Torch patch or weight cache is used. Context exception tests restored all46 relevant module identities and sys.path, and confirmed loaded model modules came from the exact owned derived directory. The context is intended for single-threaded use; concurrent global sys.modules manipulation remains outside this preflight scope.

Independent task121701_4064411349 passed complete small routed/shared expert chains (asymmetric/random and zero/extreme), dense/NZ chain, domain guards, reverse AST, module restoration and tiny mocked dual-reference adapter workflow. Additional own cases cover empty dimensions, strided operands, K131071, unsafeK131072, range-129, wrongrank, and exact odd integer83885825 above2^24 that FP32 fails to represent. Task122308_6130619024 personally reran18 model-K/domain cases and four original consumer expressions against independent INT64. This is small CPU evidence, not full40 numerical evaluation or an independently measured full-model speedup.

## Storage, budgets and staged next commands

The original store's bounded clone copies of contiguous slices, expanded tensors and0d scalars serialize compact self-contained storage; own tests inspect storage.nbytes and small file sizes. Static payload is excluded. Existing-destination/initial alias/payload corruption/owner and CPU-before-native guards pass in the tiny five-stage actual CLI run. Atomic actual+initial saving precedes either heavy reference in the source. Normal owner close and hash-bound close markers precede receipts; native positive branches/large writes were not exercised.

Own META-only contracts, with fixture builders and PreparedDispatch denied, establish:

| Component | Prefix128 | Decode1 |
| --- | ---: | ---: |
| Static logical bytes/rank | 6,463,879,104 | 6,463,879,104 |
| Static logical bytes across16 | 103,422,065,664 | 103,422,065,664 |
| Nonstatic tensor bytes P | 7,609,876,736 | 247,111,680 |
| Actual+initial logical checkpoint minimum2P | 15,219,753,472 | 494,223,360 |
| Owning exported outputs |17 |19 |

Prefix plus three positive and one stale decode checkpoint has at least17,196,646,912 logical payload bytes before serialization/container/metadata/filesystem margins. Old receipts/checkpoints are retained; budget the whole retained work directory, not only the next file. During store serialization, evaluating the next payload dict can transiently keep BOTH actual and initial cloned payload dictionaries alive, adding up to2P beyond the already-live producer actual/initial state. Static logical103GB is not a measured physical-storage/RSS number; expanded/aliased backing and reconstructed statics require address-aware accounting. Native shared IO, upload/staging, references/clones and runtime workers need separate component/lifetime accounting; do not double-count aliases or infer a full host bound from these minima. No large CPU static plan or large IO was run here.

Future concrete stage sequence remains `static-plan`, `prefix`, `validate`, `decode`, `validate-chain` on ONE fresh work root with sealed V3 dependencies and assigned-stage authorization. Before the first static-plan fixture: validate META geometry and semantic HOST/DISK/SHM budget receipt. Before prefix: source/static byte receipt, explicit ONBOARD/a2a3, per-rank free guards and effective ring config. Prefix must atomically save actual+initial bytes/ABI/source/seed identity BEFORE either reference, then close native owners. Offline validate must use both original references/direct gates with exact-S32 provenance and immutable initial oracle. Only its hash-bound accepted receipt can permit a new decode phase restoring actual prefix bytes. Decode uses one owner/artifact for128/129/130 plus stale129, checkpoints after each call, growing publication retention/page parity/parent chain, and final close. Offline validate-chain requires all19 outputs for positives and genuine original published WINDOW/C2 numerical control rejection. Command syntax/authorization schema must come from the NEW sealed V3, not unreviewed V2 bypasses. No such execution is authorized by this preflight report.

## Personal queued executions

-121054_264295690: initial manifest/source copy verification PASS.
-121701_4064411349: exact evaluator/full small expert+dense and boundary/mutable-prelude PASS.
-122205_5455915721: reviewer command failed before Python because activation path was wrong; preserved.
-122220_560176577: original toy CLI/storage/budget controls passed, then reviewer comparator scaffold TypeError; preserved.
-122308_6130619024:18 model-domain cases/four consumers/import exception closure PASS.
-122320_6222531224: corrected fresh storage/budget/growing-retention/current-row controls PASS.
-122406_6658631685: real imported META geometry/bytes17/19 PASS, no fixture or prepare.
-122519_700877405: final source/evidence verification (see final_checks.json and raw log).

No NPU job was submitted; no model/checkpoint payloads, full fixture or large IO were used. Existing native decode19/stream acceptance is preserved independently; prefix17/chain/fullweights fit/text remain OPEN. Main owns a NEW V3 integration; this report neither changes its evidence nor accepts that evolving bundle. All unexpected failures above are scaffolding/command failures and are not model failures. Reviewer PARK after final seal.
