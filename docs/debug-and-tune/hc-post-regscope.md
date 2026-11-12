# Opt-in HC post RegScope implementation

The HC operator file provides hc_post_regscope alongside the existing hc_post.
Model call sites and default --op all validation retain their implementation.
The new path is experimental and must be selected explicitly.

## Validation

The standalone entry reuses existing HC post inputs, the Torch golden, and
the atol=1e-6, rtol=1e-5 comparator:

~~~bash
python models/deepseek_v4_1_flash/hc.py --op post-regscope --batch 72 --compile-only
python models/deepseek_v4_1_flash/hc.py --op post-regscope --batch 72 -d <allocated-device>
~~~

Use normal task scheduling and the prebuilt-runtime workflow on a shared
machine. Pass --runtime-dir to reuse a compiled build and
--enable-chip-swimlane 4 for the worker timeline. Inputs are BF16 hidden
[72,5120], FP32 residual/output [72,4,5120], post coefficients [72,4],
and residual coefficients [72,4,4].

Register computation preserves ordered FP32 products and additions,
64-worker full/tail mapping, and 2560-element width tiles. It combines the
four residual rows and full-block output rows into strided DMA transfers,
keeps accumulators in registers, and stores only final values.

## Performance status

These are diagnostic measurements, not repeated end-to-end device benchmarks.
The retained C++ baseline predates the latest fused model call sites; these
numbers do not compare against the entire current model.

The implementation passed A5 hardware precision validation. In a single
same-device capture, median duration across 64 worker events was 7.404 us with
separate row DMAs and 6.037 us with batched DMAs. Both variants share UB
coefficient loads and register computation. This is not a C++ versus Python
device comparison.

Single-worker A5 dav_3510 simulation executes two full blocks and one tail
block for 72 tokens. The retained simulation used PyPTO 9b0070117,
PTOAS 0.72, CANN 9.3.0, runtime 355ce3015, and PTO ISA 5487a6b0.

| VF cycles at 1.8 GHz | Retained C++ | RegScope | Diagnostic tail barrier bypass |
| --- | ---: | ---: | ---: |
| First full block | 1050 | 1045 | 1045 |
| Second full block | 1050 | 1045 | 1045 |
| Tail block | 447 | 1813 | 351 |
| Total | 2547 | 3903 | 2441 |

The candidate retains PTOAS synchronization. The last column is a separate
object-level experiment replacing one static tail-loop SMEM_BAR.VV_ALL with
SNOP while preserving code size and hardware loop lengths. Only that
instruction word differs in the loaded replay. Its output is byte identical
to the unpatched replay for the one simulated worker. This is not a supported
execution option or a general correctness proof.

The tail coefficient index loses its [0,3] range at the VF boundary.
PTOInsertVecScopeMemBar inserts a loop-local VV_ALL. Four compile-only
constant-index controls emit no barrier. A minimized input with an explicit
unsigned remainder by four still emits the barrier, allowing address-range
analysis without relying on the caller's contract.

Tail load/execute/store triple overlap occupies 75.6% of the C++ VF, 4.3% of
the unpatched RegScope VF, and 72.6% of the diagnostic VF. These are overlapping
active intervals, not simultaneous instruction issue. The Python implementation
does not remove any automatic barrier.

A default switch requires resolving tail synchronization overhead and fresh
same-device, repeated operator and model measurements. Simulator cycles must
not be presented as device wall-time speedup.
