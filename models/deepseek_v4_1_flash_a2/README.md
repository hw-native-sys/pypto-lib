# DeepSeek V4.1 Flash on A2

Independent single-layer Prefill and A2/A3 decoder operators, with W8A8
projection, BF16 attention caches, mHC, Engram, and MoE boundaries.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a2/index.md)
for construction examples, native runtime requirements, and the precise scope
of the historical single-layer acceptance evidence.

Decoder W8A8 quantization, INT8 GEMM, layer-14/15 and L19/L20/L21/L24/L25
stages, CPU references, and validation entries are kept directly in this
directory. The prefill modules remain available through their existing entry
points; no nested `verifier` package is required.
