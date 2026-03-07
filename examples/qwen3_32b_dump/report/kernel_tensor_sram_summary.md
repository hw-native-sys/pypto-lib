# Qwen3-32B Build 生成 Kernel 的 Tensor 与 SRAM Buffer 总结

本文档基于 `qwen3-32b.py` build 后生成的 IR（`passes_dump/13_after_AllocateMemoryAddr.py`）及 `report/memory_after_AllocateMemoryAddr.txt`，对 **InCore kernel** 中的 tensor 大小与每个 kernel 的 **SRAM buffer 总大小** 进行统计与总结。

**SRAM 口径说明：** 当前按 **InCore 函数中局部变量的总大小** 计算。这些局部变量目前均为 **tensor 类型**；今后转换为 **TILE 类型** 后，即对应真实的 SRAM 占用。计算时仅统计函数体内赋值的 tensor（不含参数），按 MemRef id 去重后求和；单块 >1MB 的 buffer 视为写回参数不纳入。

---

## 1. 配置与数据来源

| 项目 | 说明 |
|------|------|
| 示例程序 | `pypto_workspace/pypto-lib/examples/qwen3-32b.py` |
| Build 输出 | `qwen3_32b_dump/`（passes_dump + report） |
| Backend | 910B_PTO |
| 关键 Pass | AllocateMemoryAddr（MemRef 已分配空间/地址） |

**模型与分块常数（与 tensor 形状一致）：**

- `BATCH=16`, `MAX_SEQ=4096`, `HIDDEN=5120`, `NUM_HEADS=64`, `NUM_KV_HEADS=8`, `HEAD_DIM=128`
- `KV_HIDDEN=1024`, `INTERMEDIATE=25600`
- `K_CHUNK=256`, `Q_OUT_CHUNK=64`, `KV_OUT_CHUNK=32`, `SEQ_TILE=120`, `MLP_OUT_CHUNK=32`, `BATCH_TILE=4`
- `CACHE_ROWS = 16*8*4096 = 524288`

**数据类型与字节数：**

- `BF16/BFLOAT16`: 2 bytes  
- `FP32`: 4 bytes  
- `INT32`: 4 bytes  

---

## 2. InCore Kernel 列表与职责

Build 后与单层 decode 相关的 **InCore kernel** 如下（不含 Orchestration 主函数）：

| Kernel 名称 | 类型 | 功能简述 |
|-------------|------|----------|
| `qwen3_decode_layer_incore_0` | 单 kernel | 输入 RMSNorm 的 sq_sum 归约（按 K chunk 累加平方和） |
| `qwen3_decode_layer_incore_1_aic` / `_aiv` | AIC/AIV 组 | Q/K/V 投影（normed_bf16 × wq/wk/wv） |
| `qwen3_decode_layer_incore_2` | 单 kernel | RoPE 应用 + K/V cache 写入 |
| `qwen3_decode_layer_incore_3_aic` / `_aiv` | AIC/AIV 组 | Decode 注意力（Q×K^T, softmax, ×V，含 online softmax） |
| `qwen3_decode_layer_incore_4_aic` / `_aiv` | AIC/AIV 组 | Attention 输出投影 + 残差（attn_out × wo + hidden_states） |
| `qwen3_decode_layer_incore_5` | 单 kernel | 后 RMSNorm 的 sq_sum 归约 |
| `qwen3_decode_layer_incore_6_aic` / `_aiv` | AIC/AIV 组 | MLP down 投影（gate*up 结果 × w_down） |
| `qwen3_decode_layer_incore_7` | 单 kernel | 最终残差加与 BF16 写回（down_proj + resid1 → out） |

Orchestration 主函数为 `qwen3_decode_layer`，内部只使用 DDR 上的大 tensor（如 q_proj、k_proj、attn_out、resid1、post_norm 等），不分配 Vec SRAM。

---

## 3. 各 Kernel 内 Tensor 形状与大小（DDR 视角）

以下按 kernel 列出 **在 kernel 内出现** 的 tensor：形状、 dtype 及由 MemRef 或形状推导的 **字节数**。同一 buffer 被多次引用时按一次计入该 kernel 的“参与计算的 buffer 集合”；大小为 0 的为 view/别名，未单独计字节。

### 3.1 qwen3_decode_layer_incore_0

- 输入/输出与中间均为 DDR。
- 无 Vec (SRAM) 分配。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| hidden_states view | [16, 128] | BF16 | 4096 |
| x_chunk (FP32) | [16, 128] | FP32 | 8192 |
| 中间 mul/row_sum | [16, 128], [16, 1] | FP32 | 8192, 64 |
| sq_sum 输入/输出 | [16, 1] | FP32 | 64 |

**该 kernel 内参与计算的 DDR buffer 规模量级：** 约 20 KB（以单次迭代内出现的最大 footprint 计）。

---

### 3.2 qwen3_decode_layer_incore_2（RoPE + K/V cache）

- 含 **Vec (SRAM) 分配**，见第 4 节。
- DDR 上为参数与 cache 的 view/assemble。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| cos_hi/lo, sin_hi/lo (view) | [1, 64] | FP32 | 0 (view) |
| k_proj view | [1, 128] | BF16 | 256 |
| k_row / k_rot | [1, 128] | FP32 | 512 |
| k_lo/k_hi 等 | [1, 64] | FP32 | 0 (view) |
| v_proj view | [1, 128] | BF16 | 256 |
| k_cache / v_cache | [524288, 128] | BF16 | 134217728（整体在 DDR） |

**该 kernel 内 DDR 上“局部”参与计算的 buffer：** 小 tensor 合计约 2 KB 量级；大 cache 为外部传入，不重复计入本 kernel 的“局部 DDR 占用”。

---

### 3.3 qwen3_decode_layer_incore_3_aic / incore_3_aiv（Decode 注意力）

- **incore_3_aiv** 含 **Vec (SRAM) 分配**，见第 4 节。
- 典型 tensor：q_rot、scores、exp、oi/li/mi、attn_row 等。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| q_rot_bf16 / q_row | [1, 128], [1, 64] | BF16/FP32 | 256, 256 |
| oi, li, mi | [1, 128], [1, 1], [1, 1] | FP32 | 512, 4, 4 |
| scores / exp_pad | [1, 64], [1, 32]~[1, 64] | FP32 | 256, 128~256 |
| k_tile / v_tile (view) | [64, 128] 或 [32, 128] | BF16 | 16384, 8192 |
| attn_row | [1, 5120] | FP32 | 20480 |

**该 kernel 内 DDR 上局部 footprint：** 约 20~40 KB 量级（与 ctx_blocks 与 tiling 有关）。

---

### 3.4 qwen3_decode_layer_incore_4_aic / incore_4_aiv（Output 投影 + 残差）

- 无 Vec 分配。
- a_chunk [16,128] BF16, w_chunk [128,64] BF16, o_acc [16,64] FP32 等。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| a_chunk | [16, 128] | BF16 | 4096 |
| w_chunk | [128, 64] | BF16 | 16384 |
| o_acc | [16, 64] / [8, 64] | FP32 | 4096 / 2048 |
| resid view | [8, 64] | FP32 | 2048 |

**该 kernel 内 DDR 局部 footprint：** 约 20~30 KB。

---

### 3.5 qwen3_decode_layer_incore_5

- 与 incore_0 类似，对 resid1 做后 RMSNorm 的 sq_sum 归约。
- 无 Vec 分配。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| resid1 view | [16, 128] | FP32 | 8192 |
| x_chunk, sq_sum 等 | [16, 128], [16, 1] | FP32 | 8192, 64 |

**该 kernel 内 DDR 局部 footprint：** 约 20 KB。

---

### 3.6 qwen3_decode_layer_incore_6_aic / incore_6_aiv（MLP down 投影）

- 无 Vec 分配。
- mlp_chunk [16,64], w_down_chunk [64,64] 或 [32,64], down_proj_acc view 等。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| mlp_chunk_bf16 | [16, 64] | BF16 | 2048 |
| w_down_chunk | [64, 64] / [32, 64] | BF16 | 8192 / 4096 |
| down_prev / down_next | [8, 64] | FP32 | 2048 |

**该 kernel 内 DDR 局部 footprint：** 约 10~20 KB。

---

### 3.7 qwen3_decode_layer_incore_7（最终残差 + 写回）

- 无 Vec 分配。
- 对 down_proj_acc 与 resid1 的 [16,64] 块做加法和 cast 写回 out。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| down_proj / resid1 view | [16, 64] | FP32 | 4096 |
| down_acc | [16, 64] | FP32 | 4096 |
| out 写回块 | [16, 64] | BF16 | 2048 |

**该 kernel 内 DDR 局部 footprint：** 约 10 KB。

---

### 3.8 qwen3_decode_layer_incore_1_aic / incore_1_aiv（Q/K/V 投影）

- 无 Vec 分配。
- normed_bf16 [16,128], wq/wk/wv chunk [128,64] 或 [64,64], q/k/v_next [8,64] 等。

| 语义 | Shape | Dtype | 大小 (B) |
|------|--------|--------|----------|
| normed_bf16_0 | [16, 128] | BF16 | 4096 |
| wq_chunk / wk_chunk / wv_chunk | [128, 64] / [64, 64] | BF16 | 16384 / 8192 |
| q_next / k_next / v_next | [8, 64] / [16, 64] | FP32/BF16 | 2048 / 2048 |

**该 kernel 内 DDR 局部 footprint：** 约 30~50 KB。

---

## 4. 每个 InCore 的 SRAM 大小（按局部 tensor 总大小）

**计算规则：**

- 仅统计 **InCore 函数体内** 赋值的 **局部变量**，且类型为 **pl.Tensor**（不含参数、不含当前已为 Tile/MemRefType 的分配）。
- 每个 buffer 按 **MemRef id 去重** 后求和；MemRef 中 `size=0` 的 view 按 `shape×dtype` 得到逻辑大小。
- 单块 **>1MB** 的 buffer 视为写回参数（如 assemble 到 k_cache/v_cache），不纳入 SRAM 预估值。

上述局部 tensor 目前均在 DDR 上分配；**今后转换为 TILE 类型并放入片上缓存后，即对应真实的 SRAM 大小**。

### 4.1 各 InCore 函数局部 tensor 总大小（函数级）

由脚本 `report/calc_incore_local_sram.py` 从 `13_after_AllocateMemoryAddr.py` 解析得到。下表为**继续优化（K_CHUNK=256, Q_OUT_CHUNK=64）后的最新 build 结果**：

| InCore 函数 | 局部 tensor 总大小 (B) | 去重 buffer 数 |
|-------------|------------------------|----------------|
| qwen3_decode_layer_incore_2_aic | 248,256 | 17 |
| qwen3_decode_layer_incore_0_aiv | 195,584 | 13 |
| qwen3_decode_layer_incore_5 | 167,424 | 5 |
| qwen3_decode_layer_incore_4_aiv | 167,168 | 6 |
| qwen3_decode_layer_incore_0_aic | 140,288 | 13 |
| qwen3_decode_layer_incore_1_aic | 140,288 | 21 |
| qwen3_decode_layer_incore_3_aic | 140,288 | 13 |
| qwen3_decode_layer_incore_3_aiv | 104,960 | 11 |
| qwen3_decode_layer_incore_1_aiv | 97,280 | 20 |
| qwen3_decode_layer_incore_2_aiv | 55,392 | 26 |
| qwen3_decode_layer_incore_4_aic | 17,408 | 8 |
| **合计** | **1,474,336** | - |

- 最大单 kernel 为 `incore_2_aic`：**248,256 B**（低于 256KB / 264KiB 上限）。
- `incore_1` 与 `incore_3` 已显著抬升：AIC 分别到 **140,288 B**（此前分别为 70,656 / 37,376）。
- 总局部 tensor 规模为 **1,474,336 B**；相较上一版（2,730,960 B）下降约 **46.0%**。

### 4.2 function_group（逻辑 kernel）并列显示（AIC / AIV）

同一逻辑 kernel 的 `_aic` 与 `_aiv` **并列展示，不做相加**：

| function_group（逻辑 kernel） | AIC 局部 tensor (B) | AIV 局部 tensor (B) | 单函数 (B) |
|-------------------------------|---------------------|---------------------|------------|
| qwen3_decode_layer_incore_2 | 248,256 | 55,392 | 0 |
| qwen3_decode_layer_incore_0 | 140,288 | 195,584 | 0 |
| qwen3_decode_layer_incore_5 | 0 | 0 | 167,424 |
| qwen3_decode_layer_incore_4 | 17,408 | 167,168 | 0 |
| qwen3_decode_layer_incore_3 | 140,288 | 104,960 | 0 |
| qwen3_decode_layer_incore_1 | 140,288 | 97,280 | 0 |
| **合计（分列）** | **686,528** | **620,384** | **167,424** |

按“你提出的目标区间 `[64KB, 264KB]`”检查：

- **下限 (<64KB)**
  - 小项已进一步减少；当前仅 `incore_2_aiv` 55,392 B 与 `incore_4_aic` 17,408 B 低于 64KB。
- **上限 (>264KB)**
  - 当前 AIC/AIV/单函数三列均 **<=264KB**；`incore_2_aic` 已降至 **248,256 B**。

### 4.3 与 Vec (Tile) 的关系

当前最新 IR 中，`incore_0_aiv`、`incore_1_aiv`、`incore_2_aiv` 含有 Vec 分配（report 中分别显示 1/1/4 个 MemRef，Limit 均为 192KB）。  
其余 InCore 的“局部 tensor”目前均在 DDR；**转为 TILE 并放入 SRAM 后，上表中的“局部 tensor 总大小”即对应该 kernel 的 SRAM 占用预估值**。

---

## 5. 说明与约定

- **DDR 大小**：来自 dump 中 `pl.MemRef(pl.MemorySpace.DDR, -1, size_bytes, id)` 或由 shape×dtype 推导；view/别名以 0 或“不重复计入”方式处理。
- **SRAM（本文口径）**：按 **InCore 函数中局部变量（tensor 类型）的总大小** 计算；不含参数，按 MemRef id 去重，单块 >1MB 不纳入。当前这些局部变量为 tensor；**今后转换为 TILE 类型后，即对应真实 SRAM 大小**。
- **Vec (Tile)**：当前版本在 `incore_0_aiv`、`incore_1_aiv`、`incore_2_aiv` 出现 `block.alloc(Vec)` / `block.load(..., target_memory=Vec)`；其 Vec Limit 在 report 中为 192 KB/kernel。
- **AIC/AIV**：同一逻辑的 AIC 与 AIV 分别统计；上表对每个 InCore 函数单独给出局部 tensor 总大小。
- 文档对应 **单层 decode、batch=16、max_seq=4096** 的 Qwen3-32B 配置；其他配置下 shape 与字节数会随之变化。

复算或更新数据可运行：`python3 report/calc_incore_local_sram.py`（需在 `qwen3_32b_dump` 上一级目录或指定 `passes_dump` 路径）。

---

*文档生成自：qwen3_32b_dump 的 13_after_AllocateMemoryAddr.py 与 memory_after_AllocateMemoryAddr.txt*
