# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode. Apply optimizations to the existing shared implementation and adapt all affected callers. |
| Deferred performance work | Prefill and decode with other token counts. Their performance need not be preserved or optimized at this stage; reconsider performance-driven branches or file splits when these become active optimization targets. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. |
