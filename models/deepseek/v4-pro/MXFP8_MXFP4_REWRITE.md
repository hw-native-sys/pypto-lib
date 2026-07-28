# v4-pro 对齐 AscendC Hybrid MXFP8–MXFP4

> 状态快照：2026-07-23。权威策略见  
> `ascendc/cann-recipes-infer/docs/models/deepseek_v4/deepseek_v4_inference_guide.md`（Hybrid MXFP8-MXFP4）。  
> 量化参考：`module/quantization/mxfp8.py` / `mxfp4.py`（Linear/MoE **block=32**，scale=`e8m0`）。

---

## 1. 当前改了什么

单算子路径已从 INT8 W8A8 代理切到真 MX / FP8。

> **2026-07-24 a5 真机实测（`task-submit --run --device auto`）**：17 个算子目前 **无一能上板跑到精度比对**——全部卡在 codegen / host-dtype / 设备数。此前「a5 compile-only PASS」的判断不再成立（`pto.tquant.mx` operand skew）。明细见下方「上板验证实测（2026-07-24）」。

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

### 上板验证实测（2026-07-25，pypto operand + simpler FP8 host-dtype 修复后复测）

本机 NPU 为 **a5**；上板目标即 `-p a5`。
已部署到主 pypto venv 的修复：① **pypto operand**——`pto.tquant.mx` 下发 4 outs（dst/scale/max/scaling）+ `quant_type`，非破坏性（DSL 仍 2-tuple）；② **simpler FP8 host-dtype**——`DataType` 加 `FP8E4M3FN/E5M2/E8M0`（跨层：`data_type.h` + bindings + `torch_interop.py`），主 venv 的 simpler 已重指向 `pypto/runtime` 并重建 ext。

**复测结论：17 个算子仍全部不过，但 compressor 那 6 个已越过 host-dtype、越过编译、跑到设备执行**，只剩 runtime 报错；tquant 那 9 个仍卡 ptoas exp-gap：

| 阻断类型 | 受影响算子 | 性质 | 补能力仓 |
|----------|------------|------|----------|
| `exp: dtype '!pto.f8E8M0' is not supported by this op yet` | expert_shared、qkv_proj_rope、mtp_projection、decode_indexer、prefill_indexer、decode_sparse_attn{_swa,_hca}、prefill_sparse_attn（9 个） | pypto operand 已补齐；卡在 **ptoas 的 `pto.tquant.mx` 不支持 e8m0 exp**（exp 位反而收 f8E4M3FN/e5m2/f32/bf16，唯独不收 e8m0）→ 该 op 在 ptoas 是半成品（0.48/0.51 均如此） | **ptoas / pto-isa**（补 exp；或 pypto 改走向量 lower，对齐 pto-isa a5 `vmax`/`vcgmax`/`vstus`） |
| `RuntimeError: run failed with code -1`（设备 runtime，**偶发**） | decode/prefill_compressor_ratio{4,128}、decode/prefill_indexer_compressor（6 个） | **simpler FP8 host-dtype 已修 + pypto operand 已修 → 这些算子已能上板跑通**：`decode_compressor_ratio4` 实测 **8/9 次 PASS**（含 FP8 `cmp_kv_cache` golden 比对）。偶发（~1/9）一次 `rtMalloc` 申请到 garbage size `0x11dfffc7af251800`（≈1.15e18B）→ OOM 207001 → code -1；属 runtime tensor-nbytes 偶发坏值（疑似内存安全/uninit/race，或 repoint 到的旧 simpler `8cdb306c` 已修于新版）。**不是确定性 simpler/pypto 故障** | simpler/pypto runtime（偶发内存安全；优先用 ASan 复现，或把 FP8 fix 移到新版 `~/Desktop/simpler` 而非 repoint 旧版） |
| `Unsupported torch dtype torch.float4_e2m1fn_x2` | expert_routed | pypto `jit/decorator.py::_torch_dtype_to_pypto` 未注册 MXFP4 packed（**pypto 侧**，非 simpler） | **pypto**（dtype 表；FP4 packed element size 与 FP8 不同） |
| `need exactly 2 devices, got [0]` | moe | EP 编排单卡不够 | 测试配置（`--device-num 2`） |

注：expert_routed 过了 dtype 后仍命中 §3 L1（设备 W4A4 vs golden W4A8）。**compressor 6 个已到设备执行**，下一步是定位 `code -1`（需 `ASCEND_PROCESS_LOG_PATH` 设备日志）；**tquant 9 个**仍等 ptoas exp 或 pypto 向量 lower。复跑脚本：`~/Desktop/board_sweep.sh`，日志：`~/board_logs/a5_*.log`。

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
- **L2（MX 上板编译）**：
  - **`pto.tquant.mx` 已打通（2026-07-25，仓 pypto）**：原 9 个走 `pto.tquant.mx` 的算子全部卡在 `pto.tquant.mx op exp: dtype '!pto.f8E8M0' is not supported`。根因：ptoas `tile_buf` 不接受 `f8E8M0`，且 `TQuantMxOp::verify` 强制 dst/exp = i8/ui8。已在 pypto 改为 **raw-bytes 模型**（dst=raw int8、exp=raw uint8，tstore 字节拷贝到 FP8 输出），并把 max/scaling scratch 做成 IR 级 tile（lowering rule 物化 `tile.create` → 内部 `tile.tquant_dps` op → codegen 发 `pto.tquant.mx`），解决 level3 下 codegen scratch 拿不到地址的问题。**`tile.tquant` 独立算子已上板 golden 对齐通过**（`tests/st/runtime/ops/test_tquant.py`，level3 编译 + a5 真机 vs pto-isa OCP golden）。
  - **新卡点：`tmatmul.mx.acc` scale shape（2026-07-25 实测）**：上述 9 算子 `f8E8M0` 错误已消失，但一致新卡在 `'pto.tmatmul.mx.acc' op expects a_scale shape to be [M, ceil(K/32)]`。根因：tquant 的 exp tile 在 Vec memory 为 32 字节对齐把 physical cols pad 到 32（`ui8 [M, K/32]` 行只 2 字节 < 32），而 matmul 的 scale 校验 physical shape 必须 = `[M, K/32]`（小 K 下两者冲突，同一 tile 不能同时满足）。pto-isa 用 TMOV 把 tquant exp（Vec flat）转成 matmul scale（LeftScale `[M, K/32]`）——pypto/lib 需补这个 scale 布局转换。补能力仓 **pypto**（reshape/move 产生 `[M, K/32]` SCALING tile）+ **pypto-lib**（算子接线）。受影响 9 算子：qkv_proj_rope / expert_shared / mtp_projection / decode_indexer / prefill_indexer / decode_sparse_attn(_swa/_hca) / prefill_sparse_attn。
  - **已验证非版本滞后（2026-07-24）**：取最新 ptoas **v0.51**（装在独立目录 `~/opt/ptoas-x86_64-0.51`，未覆盖 0.48）对同一份 `sh_up_mm.pto` 单跑，**仍报同样的 `pto.tquant.mx expects 4 or 5 operands`**。本地另一份 0.50 是 aarch64（本机不可执行）。结论：缺口在 **pypto 下发**（需补 `max`/`scaling` 两个 scratch 输出，对齐 pto-isa `TQuant(dst,src,exp,max,scaling)`），升 ptoas/pto-isa 无济于事。（已于 2026-07-25 修复，见上。）

---

## 附录：超参速查

| 场景 | group / block | scale |
|------|---------------|-------|
| Linear / MoE / o_proj / qkv | 32 | e8m0 |
| 主 KV C8 | 64 | 目标 e8m0；设备 interim FP32 |
| Indexer Q / Cache | per-token(-head) | FP32（max=448） |

禁止再用 128×128 冒充 MXFP8。
