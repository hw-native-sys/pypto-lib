# DeepSeek V4.1 Flash on A2

Independent single-layer Prefill and A2/A3 decoder operators, with W8A8
projection, BF16 attention caches, mHC, Engram, and MoE boundaries.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a2/index.md)
for construction examples, native runtime requirements, and the precise scope
of the historical single-layer acceptance evidence.

The [A5-to-A2 module alignment](MODULE_ALIGNMENT.md) is the canonical file
ownership map. It keeps A5's semantic Decode/Prefill plan visible while
preserving the A2/A3 BF16 and W8A8 ABI.

Model dimensions remain in `config.py`; the A2/A3 operator ABI and dynamic
shapes are in `operator_config.py`, while runtime TP sizing is in
`runtime_config.py`. Decoder W8A8 quantization and INT8 GEMM remain shared
leaves. Decoder paths follow the A5 file names: `decode_c2a_full.py` owns L14,
`decode_c2a_reuse.py` owns L15/L19, `decode_c1a_full.py` owns L20,
`decode_c1a_reuse.py` owns L21/L25, and `decode_c1a_reindex.py` owns L24.
`layer14_15.py` and `layers19_25.py` remain aggregate compatibility providers.
Shared SWA compute is owned by `swa_attention.py`; `decode_swa.py` remains a
compatibility export for existing callers.
