# Fused CSA QK / softmax / PV

The shared `decode_sparse_attn_csa.py` dispatches these mixed external kernels
for the fused packed-output contract. The existing statistics-producing path
remains responsible for TP > 1. Both physical AIV lanes are required.

The current kernel geometry is 64 query heads, 512 features, 128 keys per block,
one window block plus four compressed blocks, and at most 24 workers. Query
count and packed-output token stride are runtime arguments. The original
external tensor layouts, masks, BF16 probability conversion, FP32 online
softmax, sink normalization and inverse RoPE are preserved.

## Storage and dependencies

Each AIV owns 187392 bytes of UB:

| Allocation | Bytes | Lifetime |
| --- | ---: | --- |
| Running output, two 32 x 256 FP32 halves | 65536 | One query |
| Two 16 x 512 BF16 KV staging slots | 32768 | One gather chunk per slot |
| SCORE / PV / publication scratch A and B | 65536 | Explicitly ordered vector stages |
| BF16 probability / publication scratch C | 16384 | MTE3 completion before reuse |
| Indices, statistics and RoPE metadata | 6912 | Kernel / three-block ring |
| Coherent gather-index staging | 256 | One 64-row lane per block |

Each KV chunk uses one-row `TLOAD` operations into its own staging slot and is
stored before that slot is reused. Scalar UB accesses use assigned tile pointers
so the CPU simulator resolves the same storage. `MTE3 -> S` events protect the
two slot lifetimes. SCORE loads use independent UB, so KV stores do not create a storage alias with
SCORE. Softmax and PV reuse scratch A/B only after the preceding vector work.
Scratch C has a separate `MTE3 -> V` reuse event. Window and compressed gather
indices are loaded through MTE2 into dedicated UB before scalar reads;
`S -> MTE2` and `MTE2 -> S` events protect index reuse and readiness.

Cross-core events 0/1/2/3 represent KV / SCORE / probability / PV readiness.
AIC retains three 128 x 512 BF16 L1 KV slots, a 64 x 512 query, and a 64 x 128
probability tile (464 KiB total L1). PV alternates two 128 x 128 BF16 L0B
fragments, allowing the next fragment to load while Cube consumes the current
one. QK and PV preserve the two-block lead and three-slot GM transfer rings.

The AIC source is the audited PTOAS 0.72 Cube schedule with the explicit PV
L0B double buffer. The AIV source owns its UB layout and synchronization.
`kernel_entry` unpacks the existing tensor argument order and reads logical
worker and physical sub-lane identities from the runtime payload.

The runtime owns the FFTS notification base. These runtime entries must not
replace it with the tensor workspace address. Standalone simulator adapters
set their own shared FFTS base outside this runtime contract.

## Verified scope

Measured on Ascend A2/A3, 8K history, B64, S6, TP1, with two warmups and ten
measurements per group in same-card, single-process ABBA order. The mean of
the two group medians for each implementation gives an effective device
window of 576.310 -> 461.8445 microseconds (19.86% reduction) and a complete
device window of 973.375 -> 879.590 microseconds (9.64% reduction).
The host interval increases from 8805.295 to 9284.765 microseconds (5.45%);
it includes staging and validation, so this comparison claims a device-time
improvement only. These are standalone CSA measurements, including preparation kernels,
not full-model results. The two-warmup/ten-measurement protocol is an explicit
requirement for this experiment, overriding the usual CI baseline of five
warmups and 100 rounds.

Normal inputs, short windows, mixed top-k validity, all-invalid compressed
indices, block holes and causal-regression fixtures pass the device golden.
A fixed-address device regression rewrites window and compressed indices
between six mixed-kernel launches and checks each result. CPU goldens also
cover the fused path and the statistics-producing path's mixed and all-invalid
cases. Distributed TP > 1 hardware execution and the complete model were not
revalidated in this change.

The [current raw ABBA samples](../../../../docs/models/deepseek_v4_flash_dspark/csa-buffer-review-benchmark.json)
record every measurement, metric definitions, source hashes and the pinned
toolchain after coherent index staging. The
[historical samples](../../../../docs/models/deepseek_v4_flash_dspark/csa-buffer-benchmark.json)
retain the preceding implementation's measurements separately. To
reproduce this scenario, set `TP = 1`, `DP = 16` and
`DSPARK_SPEC_TOKENS = 5` in the experiment checkout configuration; these are
benchmark overrides, not changes to the deployment defaults.
