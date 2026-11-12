# DeepSeek V4.1 Flash on A2

Independent single-layer Prefill and A2/A3 decoder operators, with W8A8
projection, BF16 attention caches, mHC, Engram, and MoE boundaries.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a2/index.md)
for construction examples, native runtime requirements, and the independent A3
Decode acceptance scope. Formal Decode accuracy starts at `not_run`; no
historical Prefill or CPU result is reused.

The [A5-to-A2 module alignment](MODULE_ALIGNMENT.md) is the canonical file
ownership map. It keeps A5's semantic Decode/Prefill plan visible while
preserving the A2/A3 BF16 and W8A8 ABI.

Model dimensions remain in `config.py`; the A2/A3 operator ABI and dynamic
shapes are in `operator_config.py`, while runtime TP sizing is in
`runtime_config.py`. Decoder W8A8 quantization and INT8 GEMM remain shared
leaves. Decoder paths follow the A5 file names: `decode_c2a_full.py` owns L14,
`decode_c2a_reuse.py` owns L15/L19, `decode_c1a_full.py` owns L20,
`decode_c1a_reuse.py` owns L21/L25, and `decode_c1a_reindex.py` owns L24.
`decode_c2a_common.py` and `decode_layer_stages.py` remain shared compatibility
providers behind the canonical A5-shaped entries.
Shared SWA compute is owned by `swa_attention.py`; `decode_swa.py` remains a
compatibility export for existing callers.

Canonical A5-shaped composition entries are `decode_layer.py`, `decode_moe.py`,
`decode_indexer.py`, `prefill_fwd.py`, and `prefill_layer.py`. They select or
re-export the A2 owners without changing the W8A8/BF16 tensor ABI. The original
EP2 module paths remain available while callers migrate.

`decode_acceptance.py`, `decode_gate.py`, `decode_reference.py`, and
`decode_shape_contract.py` define the independent Decode session. The product
path is PyPTO-only. vLLM-Ascend is described as a separately queued reference
process and is never imported by product computation.
