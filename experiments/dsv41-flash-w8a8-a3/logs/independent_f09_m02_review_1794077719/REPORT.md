# Independent F09 M02 source review — FINAL/PARK

Disposition: NOT ACCEPTED for compilation/native progression without a new reviewed revision addressing R1–R3. This review inspected the immutable 76-artifact proposal only. No model import, Python execution, test, fixture, compilation, device access, payload construction, shared ledger write, or live source modification occurred. Root received the three findings immediately and authorized a separate artifact repair unit; this report does not review that future revision.

## Identity and scope

- Producer bundle: `../f09_m02_adapter_prepare_1794076190`; `artifacts.json` SHA256 `fe2f88f1680989429074557b9c3ef440a3b146db6c78fad15c942032581a87f7`; independently checked all 76 listed artifacts using shell SHA256, all OK.
- Frozen live M02 manifest: `e07b28713eadd00a306fd60264223dbf6ceec53761c7ac48c347032011c9ca25`; all 45 live paths independently checked OK. Proposed copied source45, UNAPPLIED: `011c3e0c6d79f1b0d4f47731859e5720dd94c5a08db24cfc2d3a299a2313ca9c`; all 45 copied paths independently checked OK. Patch SHA256 `7b3cabeae6583c059eb5128c04c41d4564deb402d7fc701c52e81dd0089a9c82`.
- Adapter SHA256 `8b0ad3ac5603c36a90798139c759b1ddad28e4da4841f41fbcf5d3934e52383b`. Probe SHA256 `0001e4c1d1b60da92993eca220d77d425ced44056922ab66de1ea99017dbe9c0`. Full byte pins and check transcripts are adjacent files.
- Reference archive lineage: upstream rc2 revision `8fde6d4e335fe300e2b949aab0e31659cce0811d`, as recorded by frozen HC application lineage. Independently hashed/read archived `tests/deepseek_v41_utils.py` (file SHA256 `577c667fbf8f8b6f20de8a0ac54c6c0435605aaee1f6c2ccc463b446055bd286`, `hc_post_reference` line 266 and compressor reference line 203), `vllm_ascend/models/deepseek_v41/model.py` (SHA256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`, MoE class line 151/forward line 262), and mixed_quant_attention.py (see reference_pins). Archive revision provenance is inherited, not re-fetched. No claim of full original-framework math execution.
- Read project entry/model/problem rules, A5 model development priorities, golden harness/runtime docs, L2 output and alias rules, experiment goal/coordination/reviewer brief, harness prompt/model rules/PyPTO integration. Review criteria from formatting skill applied only as criteria; no formatting edits.

## Findings

### R1 — zero-active source views admit unconditional out-of-range anchors (high; source-proven invalid access, generated/native effect untested)

`model/moe_exchange.py:409–420` creates active-length views; with middle-call ranks 8–11 active=0, `indices_active` is [0,6] and `routed_active` is [0,D]. `moe_ep16` nevertheless unconditionally reads `indices[0,0]` in `moe_reuse_wait` (line 144 vicinity) and `routed[0,0]` in `moe_return_consumed` (line 351). Those coordinates are outside the declared views. Whether lowering erases, rejects, or executes those reads has not been tested; no compiler defect is asserted.

`count_probe.py:52` also anchors export on `x_hc_out[0,0,0]`. The physical tensor is capacity 2, but zero active FFN output has no actual row producer. The read therefore cannot establish the required remote metadata publication dependency for local-active=0. `recv_meta` distributed writes and signal waits require generated producer-to-export reachability, not lexical ordering.

All ranks must continue collective count/payload/readiness/consumed protocols even when local-active=0: destination rank8 must receive and execute 96 remote routes in the middle call. Do not guard away the entire FFN. Repair must use legal physical-capacity operands or explicit task dependencies while deriving route population solely from runtime active counts. Generated zero-case task paths, metadata export reachability, and Out/InOut publication must be reviewed before any native GO. All four changed fullentry call sites (two decode, two prefill) consume this same shared operator.

### R2 — adapter computes an unconditioned reference but does not gate it for most outputs (medium/high evidence-claim defect; high confidence)

`f09_decode_case.py:293` computes `expected = _run_composed_golden(initial,2)` unconditioned. At lines 295–301, its only numerical acceptance calls use `make_fwd_comparators`. `model/decode_fwd.py:2062–2072` staged closures ignore the supplied `expected` and compare a new reference conditioned on prior actual layer snapshots; top-k closures do likewise. This is intentional existing staged behavior, but the producer REPORT/FUTURE_COMMANDS statement that BOTH references/19x3 are run as gates is inaccurate. Direct pool comparisons use unconditioned expectations; most math/cache/state outputs do not.

Keep original tolerances/staged semantics. Name the actual scoped verdict precisely; add separate unconditioned metrics or acceptance only if independently specified, never silently demand or relax a new tolerance. Candidate-conditioned reference must not be described as full unconditioned parity. A deliberate early-layer snapshot perturbation with downstream self-consistent results must remain distinguishable in the unconditioned report. F09 closure and original full-model parity are separate decisions.

### R3 — exact retention skips the untouched parity row of every active C2 state block (medium; source-proven check omission, no observed corruption)

`f09_decode_case.py:120–124` checks a complete state block only for idle requests and then continues. Active request blocks get no exact unchanged-byte check. The real compressor writes only the request's last token parity slot (`model/compressor.py:155–161`, state address block*STATE_CAPACITY+position%STATE_CAPACITY); the opposite parity slot must remain unchanged. Existing whole-state numerical gate permits an error ratio of 0.005 (`decode_fwd.py:2116` vicinity), so sparse writes to the untouched row can escape.

Add exact state retention outside the actual legal publication coordinates, including the opposite parity slot for active requests and all rows for idle requests. Keep newly written state rows on the original reference gate. A valid-address sparse poison in active-request opposite parity must reject; a legitimate current parity publication must remain allowed. Preserve existing idle-B whole block checks and growing cache history checks.

### R4 — old-physical negative receipt is weaker than its narrative (low/medium additional hardening; source-proven)

`f09_decode_case.py:358–362` treats any ValueError from the active count gate plus aggregate tensor sum=192 as the desired old-physical rejection. Wrong dtype/shape, negative lanes, or wrong source distribution can therefore be mislabeled as the intended numerical control if total remains 192. Require the old-physical result to independently satisfy the expected physical-count vector/ABI/nonnegative/destination routing before showing rejection against runtime counts. This is not evidence that a bad native result occurred. No R4 fix is implemented here.

## Sound aspects and limits

The fourfile patch preserves current HC post BF16 RINT then FP32 storage, including `_hc_post_reference`; it changes no tolerance literals, W8A8 arithmetic, loader, checkpoint, or live M02 math. Runtime token counts are passed into actual shared `moe_ffn` from every fullentry caller found; active prefixes reach HC/RMS/gate/shared FFN and routing while EP storage retains physical capacity. Source reference routing independently loops counts, including destination expert service independent of local source count. TP-peer equality and count bounds are checked by the reference. None of this proves lowering/native behavior.

The bounded geometry is internally coherent: physical T2/B2, four pages, two logical pages/request, A=[3,1], B=[0,2], C2 state A=1/B=0. Calls cover 126→127→128 with request positions advancing only after active calls, middle DP counts [1,1,0,2], global six-route populations 192→96→192. Valid-looking stale tail metadata prevents a sentinel-only implementation from passing. Physical T/B variation, EOS/slot reuse, general F09 and real-checkpoint/original-history parity remain open under goal decision9; temporary Engram-disabled decision10 does not waive them.

Ownership/lifetimes: PreparedDispatch uses one owner and static allocation set, stable shared host IO and fresh per-dispatch protocol domains. Actual poststate feeds next call; save_nonstatic clones fresh CPU contiguous storage and persists initial/actual before offline references. Offline reconstructs statics with exact byte hashes, verifies scenario metadata and actual state chain, loads candidate/initial independently, clones 12 mutable tensors inside reference, and hashes pristine inputs around both reference computations. This avoids the prior HC TensorSpec pureOut/expected-storage selfcomparison hazard. `_entry_specs` still creates tensor-valued specs only to derive a contract; this path does not invoke spec tensor creation or harness golden callbacks. Cross-name alias robustness and actual native public worker behavior have not been executed.

Retention correctly covers previously valid cache rows and idle request blocks; R3 identifies the remaining exact state-row hole. Original 19 outputs are retained; recv_meta is not invented as a wholeentry ABI output. Companion probe calls the same copied `moe_ffn` and only adds a diagnostic export, but its proof is separate from fullentry numerical acceptance.

The tiny self-evidence suite is producer affine/sentinel/control-flow evidence, not independent native/model math acceptance. This reviewer did not repeat it. No queue tasks, compile artifacts, device allocations, runtime PID evidence, model tensor payloads, or numerical results were created here.

## Recovery

Park on old76. Main's repair must be in a new directory and frozen independently. Review source/pins again before any authorization to compile. Live M02/all-active real-model preparation is a separate priority and this F09 review grants no source application, wholemodel fit, parity, quality, or hardware acceptance. Root imports the adjacent unique measured work record into shared ledgers; this reviewer made zero shared writes.
