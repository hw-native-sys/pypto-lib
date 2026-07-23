# v4-pro 对齐 AscendC Hybrid MXFP8–MXFP4

> 状态快照：2026-07-23。权威策略见  
> `ascendc/cann-recipes-infer/docs/models/deepseek_v4/deepseek_v4_inference_guide.md`（Hybrid MXFP8-MXFP4）。  
> 量化参考：`module/quantization/mxfp8.py` / `mxfp4.py`（Linear/MoE **block=32**，scale=`e8m0`）。

---

## 1. 当前改了什么

单算子路径已从 INT8 W8A8 代理切到真 MX / FP8；验收以 **a5 compile-only PASS** 为主（golden 多数已跟 MX 语义，上板数值未全量跑完）。

### 基础设施

| 产物 | 内容 |
|------|------|
| `mx_quant_common.py` | e4m3 / packed e2m1、e8m0 pack/unpack、dynamic MX quant、`BLOCK_K=32`、`MX_KV_GROUP=64`、atol/rtol 表、造数与 golden helper |
| `kv_c8_common.py` | 主 KV C8 常量（`KV_SCALE_COLS=HEAD_DIM/64` 等） |

### 已改算子（按模块）

| 模块 | 文件 | 精度落地 |
|------|------|----------|
| MoE 共享专家 | `expert_shared.py` | MXFP8 W8A8（权 e4m3+e8m0，激 dyn MX，`matmul_mx`） |
| MoE 路由专家 | `expert_routed.py` | 权 MXFP4；**设备暂 W4A4**（见 §3 L1） |
| MoE 编排 | `moe.py` | TensorSpec / 调用跟 shared+routed MX API |
| MLAProlog | `qkv_proj_rope.py` | `wq_a`/`wq_b`/`wkv` → MXFP8；激 dyn MX；`qr` 仍 INT8 留给 Indexer |
| Indexer | `decode_indexer.py`、`prefill_indexer.py`、`*_indexer_compressor.py` | `wq_b` MXFP8；`indexer_q` / Indexer Cache：FP8 e4m3 + **FP32** scale（max=448）；LI score FP8×FP8→FP32 |
| MLAEpilog o_proj | `decode_sparse_attn.py` / `_swa` / `_hca`、`prefill_sparse_attn.py` | `wo_a`/`wo_b` MXFP8 Right `[K,N]` |
| MTP | `mtp_projection.py` | `e_proj`/`h_proj` MXFP8；已删 smooth-quant |
| 主 KV C8 **写** | `decode/prefill_compressor_ratio{4,128}.py` | `cmp_kv` → FP8E4M3FN；`cmp_kv_scale` → **FP32（2^exp）interim** |
| 主 KV C8 **读** | `decode_sparse_attn.py`、`_hca`、`prefill_sparse_attn.py` | gather 后反量化→BF16 FA（存8算16） |

### 关键超参/工程妥协（已写入代码）

- Linear/MoE MX：`block_k=32`（不再用 128×128 冒充 MXFP8）。
- MX Right tile 常卡 64KB → 多处缩小 N-tile（如 qkv `QPROJ_MM_N_TILE` 512、indexer decode `MM_N_TILE` 256）。
- 主 KV C8 group=64；Indexer Cache 与主 KV **两套语义，未混用**。

---

## 2. 还有哪些没改

### 明确暂不改精度（Hybrid 也不量化 / 与 MX 无关）

- Compressor **Linear** 权重与 BF16 计算路径（仅 cache 写出改了 C8）。
- LightningIndexer **权重**、纯 BF16 LI 非 FP8 部分。
- Gate / HC / RMSNorm / RoPE tables / LMHead。
- FA 本体在反量化之后的 BF16 计算（不含 o_proj / C8 读写）。

### 计划内但尚未做完

| 项 | 说明 |
|----|------|
| `ori_kv` C8 | 滑窗 cache 仍 BF16；`decode_sparse_attn_swa.py` 无 cmp；MLAProlog / qkv **写** ori C8 未做 |
| 主 KV scale 真 E8M0 | 设备侧仍存 FP32；等 codegen / store E8M0 可靠后再切 |
| 层 / Fwd 编排 | `decode_*_layer`、`*_fwd`、`*_attention_*` 等仍旧 API，集成会编不过 |
| Step9 缺失 MX 算子 | 独立 FIA MXFP8、通用 `dynamic_mx_quant` / `swiglu_mx` 封装等未补 |
| 上板精度全量 | 多数只做了 compile-only；routed/moe 等需先对齐 golden（§3 L1）再验 |

### 造数/验收未闭环

- `expert_routed` / `moe`：设备 W4A4 vs golden 仍 W4A8（见下）。
- 层级端到端 golden / 上板对比未跑。

---

## 3. 跟 AscendC 还差什么

表中「性质」含义：

- **改不了（受阻）**：当前工具链/ codegen 做不到，或必须等外部能力；不是单纯漏改。
- **暂时没做**：代码上能改，只是还没排期或故意后置。

「补能力仓」指要打通该项时，能力缺口主要落在哪个仓库（可多仓）：

| 仓 | 角色 |
|----|------|
| **PTOAS** | 汇编/ISA 级 op（如混合 `tmatmul.mx`、E8M0 GM store / tile 扩展） |
| **pypto** | IR / codegen / runtime 接线（把 PTOAS 能力暴露成 `pl.*`） |
| **pypto-lib** | `v4-pro` 模型算子、golden、层/Fwd 编排（本仓） |

| 维度 | AscendC Hybrid 目标 | v4-pro 现状 | 性质 | 补能力仓 | 说明 |
|------|---------------------|-------------|------|----------|------|
| 路由专家 **设备** W4A8 | MXFP4 权 × MXFP8 激 | 设备 **W4A4** | **改不了** | **PTOAS** → 再 **pypto** → 最后 **pypto-lib** | PTOAS 无 FP8×FP4 混合 `tmatmul.mx`；pypto 跟 codegen；lib 再改回 W4A8 |
| 路由专家 **golden** | 与设备同精度 | golden 仍按 W4A8 | **暂时没做** | **pypto-lib** | 先把 golden 改成 W4A4 再上板验（L1） |
| 主 KV scale 真 E8M0 | E8M0 group-64 | 存 **FP32（2^exp）** | **改不了（暂）** | **PTOAS** + **pypto** → **pypto-lib** | 可靠 store/搬移 E8M0（非仅 matmul scale tile）后，lib 把 `cmp_kv_scale` 从 FP32 切回 E8M0 |
| 主 KV 打包布局 | nope+rope+内嵌 scale 等 640B 行 | 整行 512 e4m3 + 外挂 scale | **暂时没做** | **pypto-lib**（若要原生 FA 吃打包行则再动 **pypto**） | 现为功能等价存8算16；对齐 AscendC FA 输入再改 |
| `ori_kv` C8 | 与 cmp 同 C8 | 仍 BF16 | **暂时没做** | **pypto-lib** | MLAProlog 写 + SWA/CSA 读对称 |
| Indexer A8C8 | FP8 + FP32 scale | 已对齐 | — | — | 无差距（勿与主 KV e8m0 混） |
| Linear MX **单算子** | block=32 + `matmul_mx` | 主要算子已改 | — | — | 单算子侧基本齐 |
| Linear MX **层集成** | 整层接线 | 层/Fwd 仍旧 API | **暂时没做** | **pypto-lib** | 故意后置 |
| MTP 编排 | 量化投影进图 | 单算子已改、编排未跟 | **暂时没做** | **pypto-lib** | |
| FIA / 其它 MX op | 独立 MXFP8 FA 等 | 未补 | **暂时没做** | **pypto**（算子/ codegen）+ **pypto-lib**（模型接入） | Step9；算法可对照 AscendC |
| 整网 Hybrid | recipes 端到端 | 仅单算子+部分 compose | **暂时没做** | **pypto-lib**（权重/编排）；runtime 视接线再动 **pypto** | |

### 遗留跟踪

- **L1（路由专家）**：设备受阻用 W4A4（等 **PTOAS**）；golden 暂未跟设备（改 **pypto-lib**）→ 先改 golden，再等 PTOAS 回 W4A8。跟踪：`expert_routed.py` 文件头注释。

---

## 附录：超参速查

| 场景 | group / block | scale |
|------|---------------|-------|
| Linear / MoE / o_proj / qkv | 32 | e8m0 |
| 主 KV C8 | 64 | 目标 e8m0；设备 interim FP32 |
| Indexer Q / Cache | per-token(-head) | FP32（max=448） |

禁止再用 128×128 冒充 MXFP8。
