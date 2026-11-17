# C2 state precision localization — native projection matches exact dot

No model source, golden or threshold changed. compressor.py remains byte-identical
to before.py. Main owns the next full-entry actual-input capture/integration.
The full16 C2 state gate remains OPEN; this worker's controlled evidence does
not by itself accept that complete forward.

Unchanged bf16_project,actual ONBOARD task_20261107_092840_92073122209 on auto
device0: eight BF16 rows by5120,512 output rows with BF16 source-pattern weights.
The identical operands were evaluated by native code,Torch CPU FP32 matmul and
FP64 dot rounded once to FP32. The constant-weight dot also has an independent
analytic evaluation sum(x)*weight,which equals the FP64 result exactly.

Native versus exact maximum error8.9406967e-8; CPU FP32 versus exact maximum
0.0014724731. Native versus CPU FP32 fails384/4096 lanes at the existing1e-5
absolute/relative criterion. Example x=-.984375,w=.002 rounded BF16:
native/exact=-10.074462890625; CPU FP32=-10.073726654052734. Thus same-input CPU
SGEMM error exists at or above the reviewed full-forward residual scale; it is
not valid to modify native arithmetic merely to reproduce that CPU error.
Random asymmetric rows also match exact native within8.94e-8. Actual returned
operands/results are retained in projection_values.pt,projection.json.

The run normalized execution_mode=False to ONBOARD. compiled_meta.json is
platforma2a3; generated binaries are suffixeda2a3 and raw log has native Callable
images and chip.run.runner_run. The initial runner PASS skips golden; numerical
conclusions above are from separately evaluating the captured actual outputs.
No simulator result or runtime failure counts as a numerical control.

CPU task_20261107_093130_93074527942 AST-executes pinned rc2 compressor_ref;
source bytes match preserved archive. Original CPU linear uses the same FP32
SGEMM path and reproduces the discrepancy at threads1/2/8/16. Independent exact
analytic and FP64 calculations agree. This establishes a CPU evaluation error
for this controlled original mathematical projection; it is not original NPU
primitive parity. torch_npu is not installed in the active environment,so an
original torch_npu F.linear comparison was unavailable and not performed.

Proposed correction FOR REVIEW,NOT APPLIED: evaluate C2 golden projection in
FP64,then round to FP32 before storing state/performing the existing pooling.
This preserves FP32 output/BF16 operands and all current gates; does not change
candidate compute. Affected expected-value consumer is golden_c2a_compressor
through layer_reference and C2 phase/full-forward goldens. Shared C1,Engram and
index-key projection compute remains unchanged. Original CPU-bit exact helper
checks would need to distinguish mathematical oracle accuracy from CPU SGEMM
identity; old frozen artifacts must not be rewritten.

The actual full-entry stage-input capture is still required to confirm this
explains the sole residual rather than a preceding normalization/input error.
make_capture.py creates an artifact-only forward adding a BF16 Out tensor and
using its per-source slice as C2's direct normalization output buffer(no late
scratch copy/WAR). AST/py_compile pass,but full signature compile is NOT ready:
first attempt has T_DYN/t_dim conflict; reshape retry lacks inferred projection
x metadata. Both failures are preserved. Main must review/fix this diagnostic
route before any native launch. No capture/full16 kernel ran here.

Other helper failures preserved: initial syntax typo and unsupported c.module
qualified JIT call. Corrected standalone projection runs natively. These are
helper mistakes,not kernel/compiler root-cause claims or successful controls.

No shared bf16_project edit or other-file ownership transfer occurred; no model,
runtime/harness/checkpoint writes,commits,filing,resets or full weight loads.
All own jobs ended. Worker PARKED with source/evidence hashes and measured
record; Main/independent sibling acceptance remains required.

## Capture v2 addendum

After Main requested a separate diagnostic alternative, make_capture_v2.py
created capture_forward_v2.py from current decode_fwd.py. It leaves the original
C2 phase call and its attn_input buffer unchanged. After C2 full returns, an
SPMD copy stores attn_input into captured_input; a CORE_GROUP identity read/write
of x_hc_mid[0,0,0] explicitly depends on the copy, anchoring scratch lifetime
before the following FFN/next layer. The host capture is Out BF16 [16,3,T,5120].

CPU task task_20261107_094358_96441313752 signature compile PASS for a2a3/ONBOARD.
Generated rank orchestration has original attn_input as task98 Input, capture
as InOut, and task99 x_hc_mid InOut with explicit task98 dependency. This
supersedes the earlier unresolved-capture readiness only; direct-buffer v1
failures remain preserved. No native capture was run here. Main received exact
helper/binding instructions and owns the native16 actual-stage proof. All own
jobs ended; compressor.py remains unchanged. Work records contain two measured,
nonoverlapping intervals; import each unique ID once. Worker PARKED.
