# Model implementation

`models/` holds end-to-end LLM kernels, one flat directory per model build.
Files ending in `_draft.py` are work in progress and are not part of the
tracked runnable set.

| Directory | What it implements | `pypto-serving` |
| --- | --- | --- |
| [qwen3_14b](qwen3_14b/index.md) | Qwen3-14B BF16 prefill and decode with the serving contract, plus A8W8 and TurboQuant variants and the sampling components | Supported — one-card A2/A3 accuracy |
| [deepseek_v4_flash_mtp](deepseek_v4_flash_mtp/index.md) | DeepSeek V4-Flash at MTP = 1, batch 4 per card: operators, layer and MTP compositions, prefill/decode full forwards | Supported — eight-card accuracy job on relevant PRs |
| [deepseek_v4_flash_dspark](deepseek_v4_flash_dspark/index.md) | The same V4-Flash checkpoint at batch 64 per card and S = 8 DSpark speculation, with TP-sharded DSA-CP attention; under development | Not supported |
| [deepseek_v4_pro](deepseek_v4_pro/index.md) | Ascend A5 DeepSeek V4-Pro with an optional Flash preset, quantized Hybrid MXFP8-MXFP4 | Not supported |
| [qwen3_8_27b](qwen3_8_27b/index.md) | The Gated DeltaNet linear-attention layer of Qwen3.8-27B: six validated operators; the layer composition and projections are not written yet, so it is under development | Not supported |

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
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0
python models/deepseek_v4_flash_mtp/decode_layer.py -p a2a3 --ep 2 -d 0,1
```
