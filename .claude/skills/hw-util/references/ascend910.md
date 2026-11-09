# Ascend 910B/910C matmul reference

Use for A2/910B and A3/910C. The compiler target `a2a3` identifies a backend
family, not an exact hardware SKU, core count, clock, or bandwidth.

## Device scope

An A3 physical card contains two DIEs. For kernel analysis, treat each DIE as
a 910B-like compute device, but resolve its own SKU rather than assuming it
matches a particular 910B variant. CANN's
[A3 API documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnMoeUpdateExpert.md)
explicitly distinguishes the physical card from the single-DIE meaning of
its device parameters. Verify the capture's logical-device-to-DIE mapping.

- One kernel on one DIE: use that DIE's work, time, Cube peak, and HBM peak.
- Whole physical card: sum work on both DIEs over a common execution window
  and sum their compute and local-memory bandwidth peaks. An idle DIE still
  belongs in a whole-card denominator.
- Do not double one DIE's FLOPs just because it belongs to an A3 card. Do not
  sum two durations for concurrent execution. Cross-DIE timestamps need a
  common clock or a measured shared wall-time window.
- Keep HBM traffic and inter-DIE link traffic separate. Shared addressing
  does not make remote memory as accessible as local HBM at its peak rate.

## Cube reference peaks

Prefer a SKU-specific published dense Cube peak with explicit precision and
device scope. When unavailable, report a **CANN-configuration-derived Cube
reference**, with the throughput and frequency assumptions stated below.

The following values were inspected in CANN 9.3.0 platform configuration files
on 2026-11-09. Locate the selected installation's
`<CANN_ROOT>/<architecture>-linux/data/platform_config/<SoC>.ini`; installations
may use another directory layout. Check `[SoCInfo]` keys `cube_core_cnt` and
`vector_core_cnt`, and `[AICoreSpec]` keys `cube_freq`, `cube_m_size`,
`cube_n_size`, `cube_k_size`. These entries are software configuration evidence,
not measurements of the target machine or a guarantee of its maximum clock.

For an FP16 Cube throughput model of 4096 MACs per core per cycle:

```text
FLOPs/core/cycle = 2 * 16 * 16 * 16 = 8192
reference_TFLOPS = cube_core_cnt * cube_freq_MHz * 8192 / 1e6
```

Matrix instruction dimensions alone do not prove cycles per instruction. This
formula assumes the stated steady-state throughput; verify it for a different
compute mode. Do not infer another dtype's throughput only from its tile size.

| Platform | SoC | AIC / AIV per DIE | Configured MHz | FP16 Cube reference TFLOPS per DIE |
| --- | --- | ---: | ---: | ---: |
| A2 | Ascend910B1 | 24 / 48 | 1850 | 363.7248 |
| A2 | Ascend910B2 | 24 / 48 | 1800 | 353.8944 |
| A2 | Ascend910B3 | 20 / 40 | 1800 | 294.9120 |
| A2 | Ascend910B4 | 20 / 40 | 1500 | 245.7600 |
| A3 | Ascend910_9362 / 9363 | 20 / 40 | 1500 | 245.7600 |
| A3 | Ascend910_9372 | 20 / 40 | 1800 | 294.9120 |
| A3 | Ascend910_9381 / 9382 | 24 / 48 | 1800 | 353.8944 |
| A3 | Ascend910_9391 / 9392 | 24 / 48 | 1850 | 363.7248 |

Resolve the target SoC from capture metadata, `acl.get_soc_name()`, or device
inventory. Local platform files list supported devices; their presence does
not establish the installed hardware. Virtual partitions can expose fewer
resources, so state whether utilization is relative to the allocation or the
physical DIE. Recheck configuration values for the selected CANN installation.

For example, two identical 24-AIC, 1800-MHz DIEs have a combined reference of
707.7888 TFLOPS. A single-DIE kernel uses 353.8944 TFLOPS for DIE utilization.

Use the actual matrix compute path: BF16 needs confirmation of the same
throughput for the target mode; INT8 needs its own integer-operation peak
(TOPS), and FP32/other modes need their own peaks. A weight stored in 8 bits
and dequantized before FP16/BF16 matmul does not imply native INT8/FP8 compute.
Do not transfer the 950 MXFP8 peak or scale layout to A2/A3.

## HBM bandwidth evidence and MBU

Obtain the target SKU's HBM bandwidth from its hardware specification or an
explicit user-provided reference. Record whether the number is per DIE,
physical card, or server; use decimal TB/s. Do not infer HBM bandwidth from
Cube frequency, HBM capacity, or undocumented `ddr_rate` platform fields.

The [Atlas 800T A3 product page](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3)
lists eight processors, 8 x 128 GB memory, and 3.2 TB/s memory bandwidth.
This is a useful reference lead, not a universal per-DIE constant. Confirm
the bandwidth scope and target SKU in the hardware specification before use.
If 3.2 TB/s is confirmed as the sum of two identical DIEs' local HBM peaks,
the corresponding per-DIE denominator is 1.6 TB/s. That division is conditional.
It does not establish the bandwidth of every A2 SKU.

The [A2 whitepaper landing page](https://e.huawei.com/cn/documents/products/computing/b0b253c4d5da4e05ba80a39e3dc65fd2)
is another primary reference. Its PDF tables were not verified when this
reference was created; no universal A2 bandwidth default is recorded here.
If bandwidth is unknown, report effective TB/s and leave MBU uncomputed.

For an ordinary FP16/BF16 GEMM with two-byte output and no additional tensors:

```text
F = 2*M*N*K
Q = 2*M*K + 2*K*N + 2*M*N
compute_utilization_pct = 100 * F / (t_us * 1e6 * peak_TFLOPS)
effective_bandwidth_pct = 100 * Q / (t_us * 1e6 * peak_TB_s)
actual_HBM_MBU_pct = 100 * measured_HBM_bytes / (t_us * 1e6 * peak_TB_s)
```

Q is logical read-once/write-once traffic, not measured HBM traffic. Adapt it
for output precision, bias, input C, quantization scales/zero-points, temporary
buffers, or fused stages. The helper's `--scale-group` recipe describes
K-group scales on both operands; it is not a generic A2/A3 quantization format.

For whole-card MBU, sum each DIE's measured HBM bytes in the common window;
count replicated reads on both DIEs. Attribute the sum to the selected work
only with kernel-attributed measurements or an exclusive window on both DIEs;
otherwise report device-wide window utilization. Cross-DIE traffic requires identifying
which HBM controllers served it rather than blindly doubling logical bytes.

## Calculator example

This is illustrative FP16 arithmetic, not a measured A2/A3 run. It assumes a
24-AIC, 1800-MHz DIE and an independently confirmed 1.6 TB/s HBM peak:

```bash
python .claude/skills/hw-util/scripts/calculate.py \
  --m 1024 --n 1024 --k 1024 --time-us 100 \
  --time-kind profiler-duration --peak-tflops 353.8944 --peak-tb-s 1.6 \
  --a-bytes 2 --b-bytes 2 --out-bytes 2
```

Without measured traffic, the actual HBM MBU remains unknown. Omit `--peak-tb-s` if
that bandwidth assumption has not been established. For whole-card accounting
or nonstandard traffic, calculate the sums explicitly rather than pretending
the aggregate is one larger dense GEMM.
