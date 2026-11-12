# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode and attention output projection / HC-post in 4096-token prefill, including TP1 and TP>1. Apply optimizations to the shared implementation across all attention layers and adapt all affected callers. |
| Deferred performance work | Performance tuning for other token counts remains secondary to these targets; correctness and the shared fused implementation remain required. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. The optimized C++ routed FFN requires A5 device execution: the pinned CPU simulator does not model its intra-block events, shared L1 access, or dual-AIV Fixpipe transfers. Simulation requests fail explicitly; they do not select a different numerical implementation. |
