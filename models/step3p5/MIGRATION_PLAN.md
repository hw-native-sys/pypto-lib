# Step3p5 → PyPTO 迁移计划

把 `vllm/model_executor/models/step3p5.py`（Step3p5ForCausalLM）的模型实现移植到
PyPTO-Lib 的内核形式，文件组织参照 `models/qwen3/14b/`，MoE / MTP / SWA / TP
lm_head 等较重的部件参照活跃维护的 `models/deepseek/v4/`。

权威配置 / shape 来源（昇腾盘）：

```
/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16
```

---

## 一、step3p5 相对 qwen3-14B 的核心差异

| 维度 | qwen3-14B（已实现） | step3p5（要实现） |
|---|---|---|
| 层数 | 40 dense | 45 主层 + 3 MTP |
| MLP | 每层都是 dense SwiGLU | 层 0–2 是 dense；层 3–44 是 **MoE** |
| MoE | — | **288 专家、top-8、sigmoid 路由 + 学习偏置 router_bias、renormalize**，外加 1280 维共享专家 |
| 注意力 | 统一 GQA（40 头、8 KV 头） | **混合**：全注意力层 64 头、滑窗层 96 头（窗口 512），均共享 8 个 KV 头 |
| RoPE | 统一 theta、整头旋转 | **逐层** theta（全注意力 5e6、滑窗 1e4），**逐层** partial_rotary_factor（全注意力 0.5、滑窗 1.0） |
| RoPE 缩放 | 无 | llama3 yarn 缩放，**只**作用于全注意力层（`yarn_only_types=["full_attention"]`） |
| q/k norm | 普通 RMSNorm | **零中心 RMSNorm**（等价于 γ_eff = stored_γ + 1.0） |
| Attn 输出门控 | — | **逐头门控** `g_proj`：sigmoid 后逐头乘到 attn 输出上 |
| 激活 | SiLU | 大多数是 SiLU；个别层走 SwigluStep（limit=7 在 routed-MoE 的 43/44 层；limit=16 在 44 层 share-expert） |
| 投机解码 | — | 3 个 MTP 层：`eh_proj`、enorm/hnorm、每 MTP 自带 `transformer.shared_head.{norm,output}` |

---

## 二、从 checkpoint 实测出的配置

- `hidden_size = 4096`、`intermediate_size = 11264`（dense MLP）
- `num_hidden_layers = 45`、`num_nextn_predict_layers = 3`
- `vocab_size = 128896`、`torch_dtype = bfloat16`
- 全注意力层：`num_attention_heads = 64`、`num_attention_groups = 8`、`head_dim = 128`
- 滑窗层（来自 `attention_other_setting`）：`num_attention_heads = 96`、其余同上
- MoE：`moe_num_experts = 288`、`moe_top_k = 8`、`moe_intermediate_size = 1280`、`share_expert_dim = 1280`、`moe_layers_enum = "3,…,44"`
- 路由：`moe_router_activation = "sigmoid"`、`moe_router_scaling_factor = 3.0`、`norm_expert_weight = True`、`use_moe_router_bias = True`、`need_fp32_gate = True`
- step3p5 专有：`use_head_wise_attn_gate = True`、`sliding_window = 512`、`zero_centered = True`
- `rope_scaling = {rope_type:"llama3", factor:2.0, original_max_position_embeddings:131072, low_freq_factor:1.0, high_freq_factor:32.0}` 仅在全注意力层使用
- `max_position_embeddings = 262144`
- `rope_theta`（逐层）：全注意力 5_000_000.0，滑窗 10_000.0
- `partial_rotary_factors`（逐层）：全注意力 0.5，滑窗 1.0
- `layer_types`：`[full, sliding, sliding, sliding]` 4 层一循环
- `swiglu_limits`：仅层 43、44 为 7.0，其余 0.0
- `swiglu_limits_shared`：仅层 44 为 16.0，其余 0.0

---

## 三、实测的关键权重 shape

```
embed_tokens.weight                 [128896, 4096]
norm.weight                         [4096]
lm_head.weight                      [128896, 4096]                # 词表头

layer 0（全注意力、dense MLP）
  self_attn.q_proj.weight           [8192, 4096]                  # 64 头 × 128
  self_attn.k_proj.weight           [1024, 4096]                  # 8 KV 头 × 128
  self_attn.v_proj.weight           [1024, 4096]
  self_attn.o_proj.weight           [4096, 8192]
  self_attn.q_norm.weight           [128]                         # 零中心，per-head 维度
  self_attn.k_norm.weight           [128]
  self_attn.g_proj.weight           [64, 4096]                    # 逐头门控，num_heads 个输出
  mlp.gate_proj.weight              [11264, 4096]
  mlp.up_proj.weight                [11264, 4096]
  mlp.down_proj.weight              [4096, 11264]

layer 1（滑窗、dense MLP）
  self_attn.q_proj.weight           [12288, 4096]                 # 96 头 × 128
  self_attn.k_proj.weight           [1024, 4096]
  self_attn.o_proj.weight           [4096, 12288]
  self_attn.g_proj.weight           [96, 4096]

layer 3（滑窗、MoE）
  moe.gate                          [288, 4096]                   # FP32 路由（没有 .weight 后缀）
  moe.gate_proj                     [288, 1280, 4096]             # 288 个专家堆叠
  moe.up_proj                       [288, 1280, 4096]
  moe.down_proj                     [288, 4096, 1280]
  moe.router_bias                   [288]                         # FP32 学习偏置
  share_expert.gate_proj.weight     [1280, 4096]
  share_expert.up_proj.weight       [1280, 4096]
  share_expert.down_proj.weight     [4096, 1280]

MTP layer 45
  eh_proj.weight                    [4096, 8192]                  # 把 hidden 和 token embedding 拼起来再投影
  enorm.weight                      [4096]
  hnorm.weight                      [4096]
  + 标准的 self_attn / mlp / norms
  transformer.shared_head.norm.weight   [4096]
  transformer.shared_head.output.weight [128896, 4096]            # 每个 MTP 自带输出头
```

---

## 四、参考实现的对应关系（重点：deepseek-v4 优先）

仓库里已有的两个模型族就是模板。**MoE / MTP / SWA / TP lm_head 等部件优先抄
deepseek-v4**（`models/deepseek/v4/` 约 1.57 万行，持续维护），dense GQA 的
decode 主体抄 qwen3/14b。

| step3p5 要做的部件 | 最近的参考 | 改造点（相对参考） |
|---|---|---|
| dense 注意力 decode 主体（RMSNorm → QKV → RoPE → 分页 KV → flash decode → o_proj） | `qwen3/14b/decode_layer.py` + `decode_fwd.py` | qwen3 是纯 GQA，需要补：零中心 norm、partial RoPE、逐头 gate |
| 滑窗注意力 | `deepseek/v4/decode_attention_swa.py` | deepseek SWA 嵌在压缩稀疏注意力链里；step3p5 SWA 是 96 头普通窗口 GQA，去掉压缩相关逻辑 |
| MoE 门控（路由 + topk + 归一） | `deepseek/v4/gate.py` | **关键差异**：deepseek 用 group softmax topk；step3p5 用 **sigmoid** 激活，topk **前**先加 `router_bias`，topk 后 renormalize，再 × 3.0 |
| 路由专家 | `deepseek/v4/expert_routed.py` | **关键差异**：deepseek 是 INT8 W8A8 + 跨卡 EP；step3p5 是 **BF16 单卡** —— 删掉 INT8 量化 / 反量化，删掉 HCCL dispatch/combine，保留逐专家 FFN + SwigluStep |
| 共享专家 | `deepseek/v4/expert_shared.py` | step3p5 共享专家是 1280 维普通 SiLU MLP，仅层 44 用 SwigluStep(limit=16) |
| dispatch / combine | `deepseek/v4/dispatch.py` / `combine.py` | 单卡：dispatch 退化为本地 token→expert scatter，combine 退化为本地加权 gather，去掉 HCCL window |
| MTP 输入投影 | `deepseek/v4/mtp_projection.py` | deepseek 做 `e_proj(enorm(h)) + h_proj(hnorm(prev))`；step3p5 做的是把 `[enorm(h) ; hnorm(embed_next)]` **拼接**后过一个 `eh_proj [4096,8192]` |
| 最终 norm + LM 头 | `qwen3/14b/rms_lm_head.py`（单卡）或 `deepseek/v4/lm_head.py`（TP） | 先抄 qwen3 单卡；后续要 TP 再换 deepseek 模板 |
| prefill | `deepseek/v4/prefill_*` + `qwen3/14b/prefill_fwd.py` | 按 deepseek prefill 的文件拆分镜像 |

---

## 五、分阶段交付

每个阶段都产出一个可独立跑、有 torch golden 校验的产物，再进下一阶段。

### Phase 1 — 基础（已完成）

- `config.py`：所有 shape、逐层表（`LAYER_TYPES` / `LAYER_ROPE_THETA` /
  `LAYER_PARTIAL_ROTARY_FACTOR` / `SWIGLU_LIMITS` / `SWIGLU_LIMITS_SHARED` /
  `MOE_LAYER_INDICES`）、MoE 常量、动态维度占位符、tiling 默认值、`is_full_attention` /
  `is_moe_layer` / `num_heads_for_layer` / `rotary_half_for_layer` 等帮手函数。
- `MIGRATION_PLAN.md`：本文。
- 验收：在昇腾机器上 `python3 -c "import config"` 通过（已验证：
  `NUM_TOTAL_LAYERS=48, len(LAYER_TYPES)=48, MOE_LAYER_INDICES[:3]=(3,4,5),
  LAYER_ROPE_THETA[0]=5e6, LAYER_PARTIAL_ROTARY_FACTOR[0]=0.5`）。

### Phase 2 — 单层 decode draft（并行做 layer 0 / layer 1）

按用户要求两个 draft 同期交付：

- `single_layer_decode_full_draft.py` —— 对应 layer 0：全注意力、64 头、partial-rotary 0.5、
  rope_theta=5e6 + llama3 yarn、dense MLP。最贴近 qwen3/14b 的 fa_fused 模式，主要
  增量是**零中心 RMSNorm、partial RoPE、逐头 gate `g_proj`**。
- `single_layer_decode_swa_draft.py` —— 对应 layer 1：滑窗 512、96 头、partial-rotary 1.0、
  rope_theta=1e4、dense MLP。新增点是**滑窗 mask** 和 96 头分组（Q_PER_KV=12，对应
  `Q_HEAD_BATCH_SWA=12 / Q_HEAD_PAD_SWA=24` 已经在 `config.py` 里准备好了）。

两个 draft 都自带 torch golden 和 `run_jit` 入口（仿 qwen3/14b 的 `__main__`），分别
按 a2a3sim 跑通：`pass_rate ≥ 0.98`。

### Phase 3 — 把两种注意力收敛进可参数化的 decode_layer

把 Phase 2 的两份 draft 抽公共片段，做出一个能按 `layer_idx` 切换 head 数 / rotary
half / sliding window / rope_theta 的统一 attention 内核。在两个代表性的层都跑 golden。

### Phase 4 — MoE 块（288 专家 top-8、shared expert、sigmoid+bias）

参考 deepseek v4 的 `gate.py / dispatch.py / expert_routed.py / expert_shared.py /
combine.py / moe.py`，单卡版改造：
- `gate.py`：FP32 gate 矩阵 → sigmoid → 加 `router_bias` → top-8 → renormalize → × 3.0
- `dispatch.py`：单卡 token→expert scatter（BF16，不要 INT8）
- `expert_routed.py`：288 个 1280 维专家 FFN，按层选 SiLU / SwigluStep(7)
- `expert_shared.py`：1280 维共享专家，按层选 SiLU / SwigluStep(16)
- `combine.py`：单卡加权 gather
- `moe.py`：把上面 5 个拼起来，golden 跑 layer 3 输入

### Phase 5 — 多层 decode_fwd + rms_lm_head

- `decode_layer.py`：按 `layer_idx` 派发 dense-MLP vs MoE、full vs SWA、partial-rotary 因子。
- `decode_fwd.py`：loop 45 个主层 + 收尾。
- `rms_lm_head.py`：先抄 qwen3 单卡版。
- 端到端 torch golden，预期 `pass_rate ≥ 0.97`（BF16 长尾比 qwen3 的 40 层放宽一点）。

### Phase 6 — prefill

镜像 deepseek v4 的 prefill 文件拆分：`prefill_qkv_proj_rope.py /
prefill_attention_full.py / prefill_attention_swa.py / prefill_moe.py / prefill_fwd.py`，
按层类型派发。

### Phase 7 — MTP

`mtp.py`：3 个 next-n-predict 层。每层做
`hidden = eh_proj(concat[enorm(h), hnorm(embed_next)])` → 标准 step3p5 attention+MLP →
自带 `transformer.shared_head.{norm,output}` 投影到 vocab。挂到 `decode_fwd` 的尾巴上。

### Phase 8 — 全模型验收与性能

45 主层 + 3 MTP 全量跑，l2_swimlane 打开，从 checkpoint 加载真实 BF16 权重（如果到时
serving 侧已经接好 weight loader）。逐 kernel 调 tile / pipeline stage，PMU / swimlane
看瓶颈。

---

## 六、文件布局（镜像 qwen3/14b，扩展 MoE/MTP）

```
models/step3p5/
  MIGRATION_PLAN.md                     # 本文
  config.py                             # 所有 shape / 逐层表 / 动态维度
  single_layer_decode_full_draft.py     # Phase 2 — full attention 单层 draft
  single_layer_decode_swa_draft.py      # Phase 2 — sliding attention 单层 draft
  attention_full.py                     # Phase 3 — full attention 内核（按层参数化）
  attention_swa.py                      # Phase 3 — sliding attention 内核
  gate.py                               # Phase 4 — sigmoid + router_bias top-k
  dispatch.py                           # Phase 4 — 单卡 token→expert scatter
  expert_routed.py                      # Phase 4 — 288 个路由专家 FFN
  expert_shared.py                      # Phase 4 — 共享专家 FFN
  combine.py                            # Phase 4 — 单卡加权 gather
  moe.py                                # Phase 4 — MoE 总编排
  decode_layer.py                       # Phase 5 — 按 layer_idx 派发
  decode_fwd.py                         # Phase 5 — 多层 loop + lm_head
  rms_lm_head.py                        # Phase 5 — 最终 RMSNorm + 词表头
  prefill_qkv_proj_rope.py              # Phase 6
  prefill_attention_full.py             # Phase 6
  prefill_attention_swa.py              # Phase 6
  prefill_moe.py                        # Phase 6
  prefill_fwd.py                        # Phase 6
  mtp.py                                # Phase 7
```

---

## 七、验证策略

每个内核文件自带 torch golden 函数和 `run_jit` 的 `__main__`，对齐 qwen3/14b
`decode_fwd.py` 的模式。pass-rate 比较函数（`make_pass_rate_compare`）专门处理
BF16 ULP 的长尾；阈值按阶段收紧并在文件注释里写明。端到端测试等 `golden/` 目录里
落了 step3p5 入口之后再接（早期阶段以每个文件的 `__main__` 为闸口）。
