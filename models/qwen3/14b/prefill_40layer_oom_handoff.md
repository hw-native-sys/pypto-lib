# Qwen3-14B Prefill 40 层 / batch=16 / 3500 chunked —— 单卡无法跑通（问题交接）

> 环境：a2a3 / Ascend 910B 单卡，HBM **65.8GB**（实测 `torch.npu.mem_get_info`）。
> 代码：`models/qwen3/14b/prefill_fwd.py`，40 层、batch=16、context 3500、chunked prefill。

## 1. 复现命令

最能说明矛盾的一条（参数已确认生效，跑到 runtime 后 ring 溢出）：

```bash
PTO2_RING_HEAP=1073741824 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=1048576 \
python models/qwen3/14b/prefill_fwd.py -p a2a3 -d <dev> \
  --max-seq 3500 --chunk-start 2988 --chunk-size 512 --num-layers 40 \
  --enable-scope-stats
```

- `--chunk-start 2988 --chunk-size 512`：3500 上下文的**最后一块**（位置 2988~3499，前 2987 视为已在 KV cache）——最重的一块。
- 首次 golden 计算 ~11 分钟；用 task-submit/ts 时需 `--max-time 0 --timeout 3600`（否则默认 300s/600s 被超时杀掉）。
- 复用：加 `--golden-data <golden目录> --runtime-dir <同目录>` 可跳过 golden + 编译，调 ring 参数秒级完成。

## 2. scope_stats 路径

| 说明 | 路径 |
|---|---|
| **主证据**：TW=262144 + HEAP=1GB，跑到 runtime，ring3 task_window 峰值 **269,675 > 262144** 溢出 | `build_output/_jit_prefill_fwd_20260622_151714/dfx_outputs/scope_stats/scope_stats.jsonl` |
| 对照：chunk-256 + TW=131072，峰值 184,987 | `build_output/_jit_prefill_fwd_20260622_151308/dfx_outputs/scope_stats/scope_stats.jsonl` |
| chunk-512 golden 缓存（复用用） | `build_output/_jit_prefill_fwd_20260622_143639/data` |

（绝对前缀：`/data/l00955553/dev/cppcode/pypto/pypto-lib/`）

## 3. 现象

调遍 `PTO2_RING_*` 都无法跑通，被两个约束夹死：

1. **task_window 峰值 ~270k+**（ring3，热点 `prefill_fwd.cpp:172` 逐 token attention scope），需 `PTO2_RING_TASK_WINDOW=524288`（必须 2 的幂），对应静态 arena ≈ 19,614 B × 524288 ≈ **10.3 GB**。
2. batch=16 的 logits 输出 `[16, 152064] FP32 = 9.73 GB`，加 40 层权重(~30GB) + KV，把显存占满，留给 ring arena+heap 的预算只有 **~7–8 GB**（见下表）。10.3GB 装不下。

### 显存梯度（task arena + heap，实测）

| 配置 | arena + heap | 结果 |
|---|---|---|
| TW=131072 + HEAP=4GB | 2.57 + 4.0 = 6.57 GB | 装得下 → ring 溢出（峰值>131k） |
| TW=262144 + HEAP=4GB | 5.14 + 4.0 = 9.14 GB | **tensor OOM**（rtMalloc 207001, tensor 22 = 1.56GB） |
| TW=262144 + HEAP=1GB | 5.14 + 1.0 = 6.14 GB | 装得下 → ring 溢出（峰值 **269,675**） |
| TW=524288 + HEAP=1GB | 10.3 + 1.0 = 11.3 GB | 必然 OOM（超 ~7–8GB 天花板） |

## 4. 已验证的事实

- 静态 arena ≈ **19,614 B × `PTO2_RING_TASK_WINDOW`**（heap / dep_pool 另算）；两点拟合：TW=4,194,304→82.3GB、TW=8,388,608→164.5GB。
- `PTO2_RING_TASK_WINDOW` / `PTO2_RING_HEAP` **必须是 2 的幂**，否则**静默回落到编译期默认 16384**（只能从 scope_stats header 的 `task_window_max` 发现）；`PTO2_RING_DEP_POOL` 无此限制。
- task_window 峰值由 **40 层累积**决定，**与 chunk-size 几乎无关**（chunk-512 与 chunk-256 峰值同量级 ~185k–270k），所以**缩 chunk-size 不解决**。
- `PTO2_RING_HEAP` 会按比例预留 arena（scope_stats 里 heap 恒读 ~99.97%×配置值）；4GB→1GB 可腾 ~3GB，但不足以填补 10.3GB arena 的缺口。
- 错误码：ring 溢出 = AICore **507018**（device drained）；arena/张量 OOM = **rtMalloc 207001** + `bind_callable_to_runtime_impl` 失败（code -1）。

> 注：合成测试中 `chunk_lens=[0,…,0,512]`，batch=16 里仅 1 行有效，但 logits 仍按完整 `[16,152064]` 物化——显存占用由 batch **容量**决定，与有效行数无关。
