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
