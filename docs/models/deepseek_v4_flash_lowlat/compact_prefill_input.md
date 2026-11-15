# Compact prefill input operator

`CompactInput` registers mutable CPU BF16 prompt rows before Worker preparation. Fill it inside each request and retire copies before reusing it. Each rank receives the real BF16 embedding rows; `l3_expand_compact_input` writes the existing real FP32 HC allocation on device. Public model prefill inputs retain their original dtype and layout.

The device operator reconstructs FP32 bits from the two BF16 bytes and replicates them to four HC channels. Byte-sized integer intermediates are exactly representable through the pinned backend's floating cast path. This preserves NaN payloads as well as finite values. It does not reinterpret a public DeviceTensor's dtype or use placeholder allocations.

For a fixed16K prompt on TP8, embedding transfer is1GiB instead of the expanded8GiB. The operator's execution time must be included in request TTFT. CPU prompt work must still execute inside each measured request; preallocated empty shared storage is not a cached prompt.

The helper does not change target weights, quantization, cache layouts, routing, prefill arithmetic, native decode, or the1M capacity configuration. It is a separate program with two actual tensor parameters. Source/native, device bit equality, full AgentX output/cache comparisons and ordinary performance are separate qualification stages. The helper is not enabled by adding this module alone.
