# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 48- and 72-token decode for MoE return; 72-token decode for other operators. Apply optimizations to the existing shared implementation and adapt all affected callers. |
| Deferred performance work | Prefill and decode outside the active token targets. Their performance need not be preserved or optimized at this stage; reconsider performance-driven branches or file splits when these become active optimization targets. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. The optimized C++ routed FFN requires A5 device execution: the pinned CPU simulator does not model its intra-block events, shared L1 access, or dual-AIV Fixpipe transfers. Simulation requests fail explicitly; they do not select a different numerical implementation. |

## Kernel performance evidence

- [Decode MoE return: implementation, validation, and kernel timings](moe-return-kernel.md)
