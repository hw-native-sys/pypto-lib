# Experimental prefill O-projection B dispatch batching

The prefill-only `--prefill-oproj-b-batched` build option launches independent
(output slab, group) INT8 projection blocks in a single SPMD grid. The
original path submits one single-block task per slab and group. Group
quantization must finish before the batch starts, and the unchanged ordered
output epilogue waits for the entire batch.

Both paths call the same per-block compute body. The M128, N256 and K256
fragments, INT32 accumulation, scale application, group order and BF16
rounding are unchanged. Each block owns a disjoint output slab in the
existing partial tensor. No native decode implementation is changed.

The option defaults off. Source and CPU checks are separate from native,
device numerical and ordinary performance qualification. No speedup is
claimed before the full request comparison passes.

A fixed AgentX 16K replay on TP8, using two 8192-token chunks, compared
four compute/input combinations in one retained Worker. Each arm used one
warmup and four ordinary requests in balanced order. All twenty complete
output/cache hash sets matched the frozen reference on eight ranks. With
the original input path, B dispatch batching reduced the device prefill
median from 4.702 s to 4.383 s. With prepared input arenas, it reduced
4.701 s to 4.381 s.

The combined prepared-request median was 5.016 s versus 5.563 s for the
matched original path. This includes request input preparation, upload,
cache initialization, expansion and first-token readback; Worker and weight
startup and diagnostic hashes are excluded. This result does not establish
performance for other shapes, serving, decode or long-context workloads.
The option remains off by default.

The independent `--prefill-oproj-a-batched` option groups the eight A
projection N fragments in one grid per group. Every row/N-fragment block
uses the original BF16 computation, and group quantization waits for all
its fragments. It preserves the existing M128, N128 and K256 fragments.
The option defaults off and needs separate device numerical and ordinary
qualification before a speedup is claimed.

A batching was checked with the resident input lookup in a same-Worker TP8
AgentX 16K, two-8K-chunk comparison, one warm request and four ordinary requests
per arm. Request-to-first-token time, including input and cache initialization,
was 4.550466s without A batching and 4.413579s with it; all four paired samples
improved. All twenty requests in the four-arm input/compute comparison matched
the frozen full output/cache hashes. These measurements cover this opt-in
prepared-prefill configuration, not serving TPOT or long-context qualification.
