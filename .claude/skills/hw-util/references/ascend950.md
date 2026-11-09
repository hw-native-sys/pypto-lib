# Ascend 950 matmul reference

Use this table only after establishing the actual SKU/core configuration. Values
are published specifications, not device measurements or guaranteed sustained
rates. For a requested frequency-adjusted comparison, state its additional
assumptions separately from the published-peak metric.

Source: Huawei, *Ascend 950 NPU Architecture Whitepaper*, table 3-1, PDF pages
13-14. Prefer a user-supplied local copy under `build_output/` when available.
[Official PDF](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)

The whitepaper's PDF page 12 describes two AI DIEs in one 950 chip with unified
memory access. The table below and the 4-TB/s bandwidth are whole-chip
specifications. Do not halve them merely because the chip contains two AI
DIEs. Verify the logical device's exposed resources; do not transfer A3's
one-logical-device-per-DIE accounting rule to 950.

| 950DT Cube cores | BF16/FP16 TFLOPS | FP8/MXFP8/HiF8 TFLOPS | MXFP4 TFLOPS |
| --- | ---: | ---: | ---: |
| 28 | 378 | 756 | 1513 |
| 32 | 432 | 865 | 1730 |
| 36 | 486 | 973 | 1946 |

950DT published memory bandwidth is 4 TB/s. Use decimal units: TFLOPS is 1e12
FLOPs/s and TB/s is 1e12 bytes/s. The 32-AIC configuration has 64 AIVs; these do
not add to the pure Cube GEMM denominator. Cube+Vector figures such as 919
TFLOPS for this FP8 configuration are a different metric.

## DeepGEMM cross-check

[DeepGEMM-Ascend README](https://github.com/deepseek-ai/DeepGEMM-Ascend#performance)
reports 32-core 950DT GEMMs with peak denominators:

- BF16 x BF16: 432 TFLOPS.
- FP8 x FP8: 865 TFLOPS.
- FP8 x FP4: **865**, not 1730 TFLOPS.
- FP4 x FP4: 1730 TFLOPS.

The [maintainer explanation](https://github.com/deepseek-ai/DeepGEMM-Ascend/issues/1)
confirms the 32-core configuration. Reference code at revision `3a35326` uses
`2MNK` in `tests/common.py`; the README reports `bench_msprof`, CANN 9.20, cold L2.
Inspect the current local `../DeepGEMM-Ascend` revision before comparisons.
Do not treat its large-M near-peak table as a small-M acceptance threshold.
