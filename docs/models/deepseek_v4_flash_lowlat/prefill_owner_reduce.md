# Experimental prefill owner reduction

The prefill-only `--prefill-moe-owner-reduce` build option partitions each MoE slab into eight contiguous token ranges. Every source rank sends each range to its owner. The owner adds the eight FP32 partials in rank order 0 through 7, performs the original single BF16 rounding, and broadcasts the exact BF16 rows. Routing, expert arithmetic, per-token quantization, HC postprocessing, TP8 weights and cache layouts remain unchanged. Native S1/S6 decode does not use this option.

There are two real FP32 receive lanes and two real BF16 result lanes. Publication of wave i waits for local store i-1. The owner waits for all eight FP32 publishers before publishing BF16 results. Consequently every rank has retired store i-1 before any BF16 publisher of wave i proceeds. Reuse of either lane from wave i-2 is safe. Consumers wait for all eight BF16 owners. The existing two compute slots still limit preparation to two slabs in flight. Both deferred and blocking wait plans use the same epochs and ordered arithmetic.

The separate prefill-only `--prefill-moe-expert-affinity` option constructs a stable expert-ordered permutation of active physical tiles on device. Slot identities and row counts travel together. The original route mapping and combine order remain unchanged; the expert kernel receives the reordered work vectors and otherwise uses the same arithmetic. This option adds metadata work and requires its own matched performance comparison.

Both options default off. The public prefill tensor ABI remains unchanged. The BF16 result windows are actual allocations and are included even in the legacy control's memory ledger. All configured runtime rings, external pools, weights, caches and persistent communication windows must fit with the original 12 GiB reserve. Synthetic partial sums only qualify the communication operator; complete fixed AgentX 16K runs and full physical output/cache digests remain mandatory for model qualification. Logical payload reduction is a budget, not a measured latency improvement.

The prefill-only `--prefill-moe-batched-gather` option batches eight routed
tokens per copy block and sixteen shared rows per copy block. It retains
the exact route addresses, expert tiles, row scales and ordered combine.
At a 2048-token slab, the copy fan-out falls from 2049 to 384 blocks.
A fixed 16K replay with one warmup and four ordinary requests per arm
measured prefill medians of 5.211 s and 5.117 s, respectively. All complete
output/cache byte digests matched the frozen reference on eight ranks.
These measurements qualify that workload; the build option remains off
by default.

The separate `--prefill-moe-direct-post` option requires owner reduction.
It reads the completed BF16 result lane directly, applies the original
HC post multiply and ordered residual additions, and stores final token
rows. It removes the intermediate BF16 collect and HC output copy while
retaining the existing input staging, two-slot backpressure and publication
retirement. This option defaults off. In a fixed 16K replay with batched gather in
both arms, one warmup and four ordinary requests per arm measured prefill
medians of 5.112 s and 4.720 s. Prepared TTFT medians were 5.722 s and
5.340 s, including request input preparation and upload. All ten requests
matched the complete frozen output/cache digests on eight ranks. This
qualification is limited to that workload. It changes neither GEMM leaves
nor decode execution.
