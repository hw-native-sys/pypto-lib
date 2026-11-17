# C2 reference projection correction

Only compressor.py changed. golden_c2a_compressor evaluates the BF16 projection
in FP64, then rounds to FP32 immediately, before the existing pair softmax,
BF16 pooling/RMS and FP32 state writes. The shared bf16_project kernel, C1,
indexer, Engram, all thresholds and original pool/state/cast logic are unchanged.
The full AST scope check verifies the entire source after removing precisely
the authorized golden replacement, new host coverage function and its default
entry invocation against before.py.

The original pinned compressor_ref uses linear projections stored into FP32
state; its source SHA256 is
1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee.
Its preserved rc2 archive matches those source bytes. This change evaluates the
original mathematical dot more accurately on the CPU and preserves its FP32
boundary. It does not reproduce CPU SGEMM reduction error. Independent analytic
constant-weight dot, FP64 elementwise product/reduction, and actual native
projection establish the correction. The original CPU AST still reproduces
384/4096 inaccurate lanes on controlled operands; original NPU F.linear parity
is unavailable because torch_npu is not installed. This limitation remains.

Main native16 capture task_20261107_094906_98427432350 produced all48 C2 inputs,
bitwise equal to staged golden inputs, and retained returned FP32 state. This
worker re-evaluated those SAME saved inputs using corrected owning golden and
independent FP64 elementwise multiply/sum, not the candidate matmul kernel:
all corrected projections are bitwise equal to the independent oracle; actual
native state has zero bad lanes at unchanged atol=rtol=1e-5. Largest native
absolute error is1.71661376953125e-5, within the unchanged combined tolerance.
The explicit prior CPU FP32 reference reproduces all49152 bad lanes. This is a
numerical negative, not a compiler/runtime failure or threshold adjustment.
Captured input evidence SHA256:
8ec920d1b7f8b0d7c70029a70a21945b52feb4f4d42dcb77746916d67cc01b66.

Permanent owning-entry coverage validate_c2a_reference_precision runs by default.
It checks correlated BF16 dots against independent analytic D*x*w roundedFP32,
two remapped request/state blocks, untouched state rows, and even-to-odd carried
pair pooling/normalization. The same permanent function numerically REJECTS
the old FP32 reference in the artifact control. It uses no candidate compute.

Commands, all CPU through task-submit --no-device:

    source activate-pypto.sh
    python -m py_compile models/deepseek_v4_1_flash_a3_w8a8/compressor.py
    python experiments/dsv41-flash-w8a8-a3/logs/scalar_c2_streaming_1794014676/worker_c2_referencefix/check.py

First CPU task_20261107_100236_12625326616 exit0; final extended control/scope
check task_20261107_100321_126538713370 exit0. final_queue.log/results.json retain
exact results. No failed new test; archived original worker failures/logs remain
unchanged. No NPU job submitted by this resumed worker. Main owns production
native16 all-output, continuation and independent acceptance. This CPU evidence
does not close those gates. No full weights, runtime/harness/checkpoint writes,
commits or external filing. Worker PARKED after final sealing.
