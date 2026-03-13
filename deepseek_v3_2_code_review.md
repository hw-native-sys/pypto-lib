# DeepSeek v3.2 代码审查问题报告

**审查日期**: 2026-03-09  
**审查人**: AI Assistant  
**审查范围**: deepseek_v3_2 的 4 个 Python 代码文件 (prefill/decode front/back)

---

## 🔴 严重问题

### 1. API 兼容性问题 - `pl.view` 已移除

**影响文件**: 全部 4 个文件
- `deepseek_v3_2_decode_front.py` (38 处)
- `deepseek_v3_2_decode_back.py` (14 处)
- `deepseek_v3_2_prefill_front.py` (36 处)
- `deepseek_v3_2_prefill_back.py` (14 处)

**问题描述**: 
代码使用已废弃的 `pl.view(tensor, shape, offsets)` API，当前 pypto 版本已移除该操作。

**错误信息**:
```
Error: Unknown tensor operation: view
```

**修复方案**: 
批量替换 `pl.view(` → `pl.slice(` (签名相同，可直接替换)

**状态**: ✅ 已完成替换

---

### 2. Tile/Tensor 类型混用 - 无法运行

**影响文件**: 全部 4 个文件

**问题描述**: 
pypto 新 API 严格区分 Tile 和 Tensor 类型：
- Tile ops (`pl.rsqrt`, `pl.matmul`, `pl.row_expand_mul`, `pl.col_expand_mul`) 需要 **TileType** 输入
- Tensor ops (`pl.assemble`, `pl.add`, `pl.slice`) 操作 **TensorType**
- 当前代码直接将 Tensor 传给 Tile ops，导致类型错误

**典型错误**:
```
Error in tile operation 'rsqrt': The operator tile.rsqrt requires argument to be a TileType, but got TensorType
  --> deepseek_v3_2_decode_front.py:179:26
    |
179 | inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
    |           ^^

Error in tile operation 'row_expand_mul': The operator tile.row_expand_mul requires first argument to be a TileType, but got TensorType
  --> deepseek_v3_2_decode_front.py:191:55
    |
191 | normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
    |                                                        ^^
```

**修复方案**: 
需要系统性修改代码模式：
```python
# ❌ 错误模式
x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
normed = pl.row_expand_mul(x_chunk, inv_rms_tile)  # x_chunk 是 Tensor

# ✅ 正确模式
x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
x_chunk_tile = pl.load(x_chunk, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
normed = pl.row_expand_mul(x_chunk_tile, inv_rms_tile)  # x_chunk_tile 是 Tile
```

**累加器也需要修改**:
```python
# ❌ 错误模式
q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
q_acc = pl.add(q_acc, pl.matmul(...))  # matmul 返回 Tile，add 需要 Tensor

# ✅ 正确模式
q_acc_tile = pl.create_tile([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
q_acc_tile = pl.add(q_acc_tile, pl.matmul(...))  # 都是 Tile
q_acc = pl.store(q_acc_tile, [0, 0], pl.create_tensor(...))  # 转回 Tensor
```

**预计工作量**: 每个文件约 50-100 处修改

**状态**: ⏸️ 待修复

---

### 3. InCore 内存使用异常 - Vec 利用率 0%

**影响文件**: `deepseek_v3_2_decode_front.py`, `deepseek_v3_2_prefill_front.py`

**问题描述**: 
编译器 Pass08 (AllocateMemoryAddr) 报告显示 InCore Vec 空间利用率几乎为 0%：

```
=== Memory Usage Report ===
Pass: AllocateMemoryAddr
Backend: 910B_CCE

--- deepseek_v3_2_decode_front_layer_incore_0_aiv ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Vec    |       16 B  |   192.0 KB  |    0.0%  |  1

--- deepseek_v3_2_decode_front_layer_incore_2_aiv ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Vec    |        0 B  |   192.0 KB  |    0.0%  |  4
```

**可能原因**:
1. 代码使用 `pl.auto_incore()` 但没有显式使用 `pl.load(..., target_memory=pl.MemorySpace.Mat)` 指定数据加载到 InCore
2. 编译器无法自动推断哪些 Tensor 应该放在 InCore 中
3. `pl.slice()` 返回的是 GM (Global Memory) Tensor，不是 InCore Tile

**修复方案**: 
在 `pl.auto_incore()` 范围内，所有需要高性能访问的数据应该：
```python
# 显式加载到 InCore
data_tile = pl.load(data_tensor, [0, 0], [M, N], target_memory=pl.MemorySpace.Mat)

# 或者使用 create_tile 创建 InCore 临时变量
acc_tile = pl.create_tile([M, N], dtype=pl.FP32)
```

**状态**: ⏸️ 待修复 (与问题 2 一起解决)

---

## 🟡 中等问题

### 4. 代码结构问题 - Scope 划分

**观察**: 
- `decode_front.py` 使用 2 个 `pl.auto_incore()` scope
- `prefill_front.py` 使用 1 个 `pl.auto_incore()` scope
- `decode_back.py` 和 `prefill_back.py` 结构不同

**建议**: 
统一 4 个文件的 scope 划分策略，确保：
1. RMSNorm + 投影 在一个 scope
2. RoPE + Attention 在另一个 scope (如果需要)
3. 每个 scope 内的数据流清晰，便于编译器优化

---

## 🟢 良好实践

### 5. 容量预算检查 - PASS

**文件**: `deepseek_v3_2_*_front_capacity_budget.md`

两个 front 文件的容量预算都通过：
- `decode_front`: peak 151552 B / 163840 B = 92.50% ✅
- `prefill_front`: peak 153600 B / 160 KB = 93.75% ✅

---

## 📋 修复优先级

| 优先级 | 问题 | 影响 | 预计工时 |
|--------|------|------|----------|
| P0 | Tile/Tensor 类型混用 | 代码无法运行 | 4-6 小时 |
| P0 | `pl.view` API 替换 | 代码无法运行 | ✅ 已完成 |
| P1 | InCore 内存使用优化 | 性能严重下降 | 与 P0 一起解决 |
| P2 | 代码结构统一 | 可维护性 | 2 小时 |

---

## 📝 下一步行动

1. **立即**: 修复 Tile/Tensor 类型问题 (P0)
   - 修改 `pl.load()` / `pl.store()` / `pl.create_tile()` 模式
   - 测试运行 `deepseek_v3_2_decode_front.py`

2. **验证**: 运行编译器 pass，检查 InCore 内存报告
   - 确认 Vec 使用率提升到合理水平 (>50%)

3. **统一**: 审查 4 个文件的 scope 划分，确保一致性

4. **文档**: 更新 kernel flow analysis 报告

---

## 🔗 相关文件

- 源代码: `/data/z00885570/pypto3.0/pypto-lib/examples/deepseek_v3_2_*.py`
- Pass Dump: `/data/z00885570/pypto3.0/pypto-lib/examples/deepseek_v3_2_*_dump/passes_dump/`
- 报告: `/data/z00885570/pypto3.0/pypto-lib/examples/deepseek_v3_2_*_dump/report/`
