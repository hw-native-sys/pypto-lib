# Model implementation

`models/` holds end-to-end LLM kernels, one flat directory per model build.
Files ending in `_draft.py` are work in progress and are not part of the
tracked runnable set.

| Directory | What it implements | `pypto-serving` PR CI in this repository |
| --- | --- | --- |
| [qwen3_14b](qwen3_14b/index.md) | Qwen3-14B BF16 prefill and decode with the serving contract, plus A8W8 and TurboQuant variants and the sampling components | One-card A2/A3 accuracy |
| [deepseek_v4_flash_mtp](deepseek_v4_flash_mtp/index.md) | DeepSeek V4-Flash at MTP = 1, batch 4 per card: operators, layer and MTP compositions, prefill/decode full forwards | Eight-card accuracy |
| [deepseek_v4_flash_dspark](deepseek_v4_flash_dspark/index.md) | V4-Flash with seven DSpark draft tokens and TP-sharded DSA-CP attention. At TP4, the group verifies up to 64 requests per step (16 per rank). | A3 eight-card EP8, K=7 accuracy |
| [deepseek_v4_flash_lowlat](deepseek_v4_flash_lowlat/index.md) | Batch-one V4-Flash decode and 43-layer chunked prefill with CP8 attention and TP8 MoE and LM head | No model-specific job |
| [deepseek_v4_1_flash](deepseek_v4_1_flash/index.md) | A5 prefill and decode with full attention weights on each rank, configurable prefill CP and MoE EP, independently sharded LM head, and DSpark | No model-specific job |
| [deepseek_v4_1_flash_a3](deepseek_v4_1_flash_a3/index.md) | DeepSeek V4.1 Flash W8A8 Prefill and Decode with resident weights and paged caches on 16 A3 devices (DP4/TP4/EP16) | No model-specific job |
| [glm5_3_flash](glm5_3_flash/index.md) | W8A8 prefill and decode on A2/A3 at TP8/EP8 or TP16/EP16, with hybrid KDA and NoPE-MLA attention, MoE, MTP, and a vision encoder | No model-specific job |
| [qwen3_8_27b](qwen3_8_27b/index.md) | W8A8 Gated DeltaNet block with checkpoint-compatible projections and per-request convolution and recurrent state for packed prefill | No model-specific job |

Each page covers that directory's deployment configuration and how its files
compose. The Qwen3-14B and V4-Flash MTP pages also carry the optimization
history of their tuned path —
[Qwen3-14B optimization](qwen3_14b/optimization.md) and
[V4-Flash decode optimization](deepseek_v4_flash_mtp/decode_optimization.md) —
which record which levers moved the number and what each one cost.

Entry points take script-specific platform and device arguments; inspect
`--help`, the [platform table](../get-started/first-kernel.md#platforms-and-devices), and the
[Golden Harness guide](../run-and-validate/golden-harness.md).

```bash
python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0
python models/deepseek_v4_flash_mtp/decode_layer.py -p a2a3 --ep 2 -d 0,1
```
