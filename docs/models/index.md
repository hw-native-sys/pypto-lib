# Model implementation

`models/` holds end-to-end LLM kernels, one flat directory per model build.
Files ending in `_draft.py` are work in progress and excluded from CI.

| Directory | What it implements | `pypto-serving` |
| --- | --- | --- |
| [qwen3_14b](qwen3_14b.md) | Qwen3-14B BF16 prefill and decode with the serving contract, plus A8W8 and TurboQuant variants and the sampling components | Supported — one-card A2/A3 accuracy job on relevant PRs |
| [deepseek_v4_flash_mtp](deepseek_v4_flash_mtp.md) | DeepSeek V4-Flash at MTP = 1, batch 4 per card: operators, layer and MTP compositions, prefill/decode full forwards | Supported — eight-card accuracy job on relevant PRs |
| [deepseek_v4_pro](deepseek_v4_pro.md) | Ascend A5 DeepSeek V4-Pro with an optional Flash preset, quantized Hybrid MXFP8-MXFP4 | Not supported |
| `deepseek_v4_flash_dspark` | The V4-Flash operators re-sized to batch 64 per card and S = 8 DSpark speculation; under development, operators only | Not supported |
| `deepseek_v3_2` | DeepSeek V3.2-EXP as a front/back split of one layer | Not supported |
| `qwen3_32b` | Qwen3-32B single-layer decode in two tensor layouts | Not supported |

The linked directories have a page covering their deployment configuration and
how their files compose. The rest are kernel harnesses validated against the
Golden Harness only.

Entry points take script-specific platform and device arguments; inspect
`--help`, the [platform guide](../get-started/platforms.md), and the
[Golden Harness guide](../run-and-validate/golden-harness.md).

```bash
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0
python models/deepseek_v4_flash_mtp/decode_layer.py -p a2a3 --ep 2 -d 0,1
```
