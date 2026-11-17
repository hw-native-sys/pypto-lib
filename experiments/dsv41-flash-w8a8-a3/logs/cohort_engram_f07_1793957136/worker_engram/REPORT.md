# F05 Engram worker delivery

Only engram.py and this artifact directory changed. Main owns all caller,
loader and F06 ordering integration. ABI is
engram_from_rows(lookup,wkv_weight,weight,rotation,x,output), where stored R is
FP32[32,32], shared across every32-channel hidden block. External x/output are
FP32. Host fixture_rotation() exposes the independently validated nonidentity
orthogonal block for Main fixtures.

The gate now restores hidden FP32 with h_rot @ R.T for original-basis norms and
q*k/key dot products. It uses vector32 FP32 reductions; no BF16 cube narrowing
is inserted into restoration. Keys and values remain those projected from lookup.
V already has forward rotation folded, so its residual update stays in the
rotated basis. The signed square-root clamp now uses a sign comparison instead
of dot/max(abs(dot),1e-6), preserving the original positive-zero behavior.

Pinned official rc2 common.py and layer.py are AST-executed by check.py. The
original layer calls a ReplicatedLinear on BF16 lookup/BF16 FLOAT checkpoint
wkv; the CPU stand-in uses BF16 F.linear and returns BF16 before gate.float().
The shared projection now likewise narrows BF16 before widening key/value in
the gate. Original gate returns to hidden.dtype, so the assigned FP32 HC ABI
requires FP32 residual output; the candidate unconditional BF16 round is removed.
This is source-backed BF16-linear/FP32-hidden semantics with CPU primitive
stand-ins, not original torch_npu hardware/full-checkpoint parity. Parent must
retain original global reduction/quantization limitations.

Permanent defaults exercise the real shared operator: randomT8, signed sparse
T1/T7, zero-dot sparseT1, strict cast-sensitive randomT3 seed1907. The sparse
golden independently derives a one-column exact projection and paired inverse
rotation without invoking normal candidate golden/compute. Original forward/
gate CPU execution matches normal and analytic references T1/T7 at1e-6.
Strict sparse gates are1e-4/no outliers; strict random gate1e-4/0.5% outliers;
the existing ordinary1e-2/1/128/0.5% gate is unchanged.

Final device sweep task_20261106_174208_14992512691 uses automatic device0,
run.sh/final_v2. Six actual numerical controls must reject output: frozen old
operator,missing basis,wrong transpose,double-rotated V payload,FP32 projection,
BF16 output round. Exceptions do not satisfy controls. Corrected original-source
sparseT7 and strict randomT3 pass, then all five permanent default cases pass.
Seven positive executions repeat some defaults; they are not seven distinct
fixture designs. summary.json records each failing-lane count/max difference.

Earlier own-code failures are preserved: TensorType versus tile.read mismatch,
missing row_sum temp tile, and tile store via tensor subscript. Initial FP32
projection mutation retained redundant FP32 casts and failed compilation; the
corrected mutant removes those casts and is numerically rejected. No failed
compile is counted as a negative control. Queue jobs/tasks/logs are retained.

Fresh CPU run: activate environment then check.py host with ENGRAM_EVIDENCE_DIR
set to a fresh directory, under task-submit --no-device. Fresh NPU run:
task-submit --device auto --max-time1800, execute run.sh /absolute/fresh/output
from repo root. Script activates environment and pinned PTOAS itself. Source
and reference hashes/work records are in summary.json/manifest.json. Main
imports the unique work record once; no shared ledger edits by this worker.

Worker parks after final evidence collection, no source writes or own jobs
remain. This is Main-owned worker self-evidence, not independent acceptance.
No F06/order,real-loader,full-forward,16-device,checkpoint/generation or inactive-
row lifecycle proof is claimed. No simulator/commits/external filing/protected-
source writes. Main must finish caller ABI integration before using this file.
