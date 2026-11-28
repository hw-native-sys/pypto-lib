# DeepSeek V4.1 Flash A3 Prefill and Decode

Generation uses separate 40-layer Prefill and Decode graphs on 16 A3 devices
(DP4/TP4/EP16). Both use resident W8A8 weights and the same paged history.
Prefill supports six attention modes. Decode has separate encoder and decoder
schedules. Packed sessions accept multiple live requests per DP group and
preserve each request's cache between dispatches.

See the [model entry points](../../../models/deepseek_v4_1_flash_a3/README.md)
for compilation and usage and the [accuracy guide](accuracy.md) for comparison
against a fixed vLLM-Ascend token history.

## Development priorities

- Performance target: run one distributed dispatch per prompt chunk and one
  per subsequent token, reusing rank-local weights, communication windows
  and cache allocations.
- Compatibility requirements: preserve the W8A8 checkpoint layout, six
  Prefill attention modes, Decode layer roles, DP4/TP4/EP16 ownership and
  the seven shared cache buffers.
- Deferred performance work: measure end-to-end timing and tune kernels on
  an allocated 16-device group.
