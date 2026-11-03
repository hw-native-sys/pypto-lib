# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode. Apply optimizations to the existing shared implementation and adapt all affected callers. |
| Deferred performance work | Prefill and decode with other token counts. Their performance need not be preserved or optimized at this stage; reconsider performance-driven branches or file splits when these become active optimization targets. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. The optimized C++ routed FFN supports A5 device execution and CPU simulation. CPU simulation uses TPipe FIFOs to transport converted weights, projection results, activation payloads, and MX scales between private core memories while retaining the device path's GEMMs and weighted SwiGLU quantization. The fused index scorer simulates BF16 per-head FMAs and stable logical-order TopK selection, including paged and gathered validity and short histories. |
