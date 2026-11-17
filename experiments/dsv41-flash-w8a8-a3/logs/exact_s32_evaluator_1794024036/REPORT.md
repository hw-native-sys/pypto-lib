# Isolated exact-S32 CPU evaluator prototype

The copied evaluator is ready for independent CPU review, not production/native
acceptance. ROOT45 ab1, both frozen checkpoint helpers, worker evidence and the
isolated generation patch remain unchanged. No NPU, actual checkpoint payload,
full40 fixture/reference run, large IO or full-weight allocation occurred.

`evaluator.py` builds a fresh copy of all45 frozen model modules after checking
every source hash. It whitelists exactly one expression in each of four named
function scopes: moe_experts._gemm_golden, gemm_w8a8.golden_w8a8_gemm,
mla_proj._golden_mla_proj.gemm, indexer.golden_index_scores. Only the matmul
callee changes; original .to(INT32) arguments, FP32 scaling, BF16 casts and all
surrounding operations remain intact. Reverse-transformation AST equality is
checked for the entire four modules, and the other41 copies are byte-identical.
No global Torch monkeypatch or converted-weight cache is installed.

The checked primitive requires CPU rank2 compatible INT8/INT32 tensors, actual
signed-INT8 values, and K<=131071. It rejects unsupported cases before the dot;
there is no FP32 approximation or altered INT32 overflow behavior. Integer
products/subtotals fit both S32 and FP64's exact-integer range. See the separately
sealed cpu_s32_readiness_1794023355/MATH.md and its18-case/INT64 proof, four
consumer expression checks, above2^24 negative and matched-interface benchmark.
The optional FP64 operand conversions are transient for each dot, not a cache.

`reference_context()` imports copied full-entry/reference modules under a
single-threaded CPU context, verifies source/library provenance and restores
preexisting Python module identities/path afterwards. `validate_checkpoint_values()`
is the wired CPU entry: supplied already-verified actual/initial checkpoint
values -> original unconditioned40 reference -> original comparator workflow
with lazy staged40 reference -> unchanged owning thresholds/direct gates and
unconditioned logit delta. Prefix17/decode19 schemas are enforced. It records
an explicit derived-evaluator receipt and never grants native resume permission.
The checkpoint adapter must still verify copied bytes, source/ABI/owner/static
reconstruction and close receipts before calling this evaluator. Its original
RealProvider intentionally rejects derived module paths; a separately reviewed
new adapter is needed to bind this explicit receipt, not bypass that guard.

CPU task120242_603226043 passed the first full-module differential. Final
120334_867617450 rechecks the updated
wired evaluator. Four complete expert chains (routed/shared, asymmetric/random
and zero/extreme rows) are bitwise equal through quantization, NZ unpacking,
gate/up, clipping/SiLU, optional nonuniform route scaling, BF16 cast,
requantization and down projection. A complete asymmetric dense/NZ chain is
bitwise equal. Domain/dtype/overflow and derived-source corruption reject;
import identities restore. The full-entry modules import, but no full40
numerical result is claimed. Adapter two-reference/threshold/schema control
flow uses explicitly CPU-MOCKED tiny modules for both phases.

Initial task120214_50508736 failed before model import because the command
omitted activate-pypto.sh and loaded an incompatible system libstdc++. Raw
failure is retained; the supported activation fixes the command only. Initial
successful source/receipt/results are retained separately before adding the
explicit validation entry. All own jobs ended; no native execution was attempted.

The fair benchmark belongs to the separate proof artifact: CPU16 threads,
M128/N64/K5120, original INT32 argument conversions retained on both variants,
2 warmups/9 alternating samples:3.902ms S32 vs0.803ms checked FP64->S32 median,
4.861x for this dot only. Full offline40-layer acceleration and peak RSS remain
unmeasured. Independent CPU review should inspect whitelist/provenance and
re-run bounded differential tests before any future actual-byte capture /
offline evaluation / decode resume assignment. No inline native-prefix repeat.
