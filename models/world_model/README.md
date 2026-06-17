# Fun-Control 1.3B World Model — pypto3.0 Sub-Networks

本目录包含 Fun-Control 1.3B 世界模型的 5 个 pypto3.0 子网络实现，对应 `infer_fun_control_1_3b_text.py` 推理 pipeline 中 `WanVideoPipeline` 内部的各个计算阶段。

## Pipeline 总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Fun-Control 1.3B Inference Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │  Text Prompt │───→│ T5 Encoder   │───→ text_context [1, T5_SEQ, T5_DIM] │
│  └──────────────┘    └──────────────┘                                       │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │ Reference    │───→│CLIP Encoder  │───→ clip_context [1, CLIP_TOKENS,     │
│  │ Image        │    └──────────────┘              CLIP_DIM]                │
│  └──────────────┘                                                            │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │ Images       │───→│ VAE Encoder  │───→ latent [1, 16, LAT_F, LAT_H,     │
│  │ (ref/input/  │    └──────────────┘              LAT_W]                   │
│  │  control)    │                                                            │
│  └──────────────┘                                                            │
│                                                                              │
│                    ┌──────────────────────────────────────────────────────┐  │
│                    │              DiT Forward (Denoising)                 │  │
│                    │  latent + noise + timestep + text_context +          │  │
│                    │  clip_context → denoised_latent                      │  │
│                    └──────────────────────────────────────────────────────┘  │
│                                                                              │
│                    ┌──────────────┐                                         │
│                    │ VAE Decoder  │───→ video_frames [1, 3, T, H, W]      │
│                    └──────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 子网络详解

### 1. T5 Encoder (`t5_encoder.py`)

**职责**: 将文本 prompt 编码为语义特征向量，为 DiT 提供文本条件。

**架构**:
```
RMSNorm → Self-Attention (with relative position bias) → Residual →
RMSNorm → Gated GELU-tanh FFN → Residual
× T5_LAYERS layers
→ Final RMSNorm
```

**输入**:
- `x_in`: [T5_SEQ, T5_DIM] — already-embedded text features (FP32)
- `weights`: T5 encoder weights (layer weights, position bias, final norm)

**输出**:
- `text_context`: [T5_SEQ, T5_DIM] — encoded text features (BF16)

**Pipeline 位置**: 
- 编码 `prompt` 和 `negative_prompt`
- 输出作为 DiT 的 text conditioning input

**Golden Case 配置** (config.py):
- T5_DIM = 128, T5_SEQ = 32, T5_LAYERS = 1
- Real case: T5_DIM = 4096, T5_SEQ = 512, T5_LAYERS = 24

---

### 2. CLIP Encoder (`clip_encoder.py`)

**职责**: 将参考图像编码为视觉特征向量，为 DiT 提供图像条件。

**架构**:
```
PatchConv2d → CLS + PosEmbed → PreLN →
[LN → MHA → Proj → Res → LN → FFN(QuickGELU) → Res] × CLIP_LAYERS
```

**输入**:
- `img_flat`: [1, IMG_FLAT_SIZE] — flattened reference image (IMG_FLAT_SIZE = 3 * CLIP_IMG * CLIP_IMG)
- `weights`: CLIP encoder weights (patch conv, cls, pos embed, layer weights)

**输出**:
- `clip_context`: [CLIP_TOKENS, CLIP_DIM] — encoded image features (BF16)

**Pipeline 位置**:
- 编码 `reference_image` (GT first frame)
- 输出作为 DiT 的 image conditioning input

**Golden Case 配置** (config.py):
- CLIP_DIM = 128, CLIP_TOKENS = 5, CLIP_LAYERS = 2
- Real case: CLIP_DIM = 1280, CLIP_TOKENS = 257, CLIP_LAYERS = 31

---

### 3. VAE Encoder (`vae_encoder.py`)

**职责**: 将视频帧编码到潜在空间，为 DiT 提供 latent 表示。

**架构**:
```
CausalConv3d(3→96) →
Stage 0: ResBlock(96→192) + ResBlock(192) + Downsample → [1, 192, 3, 32, 32]
Stage 1: ResBlock(192→384) + ResBlock(384) + Downsample → [1, 384, 2, 16, 16]
Stage 2: ResBlock(384→768) + ResBlock(768) + Downsample → [1, 768, 1, 8, 8]
Stage 3: ResBlock(768→1536) + ResBlock(1536) → [1, 1536, 1, 8, 8]
Middle: ResBlock + Attention + ResBlock → [1, 1536, 1, 8, 8]
CausalConv3d(1536→32) → chunk → mu [1, 16, 1, 8, 8]
→ scale_norm → output
```

**输入**:
- `video`: [1, 3, T, H, W] — video frames (reference/input/control)
- `weights`: VAE encoder weights

**输出**:
- `latent`: [1, 16, LAT_F, LAT_H, LAT_W] — encoded latent representation

**Pipeline 位置**:
- 编码三类视频: `reference_image`, `input_image`, `control_video`
- 输出 latent 作为 DiT 的输入

**Golden Case 配置** (config.py):
- VAE_Z_DIM = 16, H = W = 64, CHUNK_SIZE = 5
- LAT_F = 1 (encoder natural output: 5→3→2→1), LAT_H = 8, LAT_W = 8

---

### 4. DiT Forward (`dit_forward.py`)

**职责**: 扩散 Transformer 核心，在潜在空间执行去噪过程，生成新的 latent。

**架构**:
```
ref_conv + patch_conv + concat → x_full
text_proj(GELU-tanh) + clip_proj(LN→FC→GELU→FC→LN) → ctx
AdaLN → Self-Attn(RoPE) → Cross-Attn(text+img) → FFN → Head → output
```

**输入**:
- `x_input`: [1, DIT_IN_DIM + COND_CH, FP, LAT_H, LAT_W] — noisy latent
- `text_raw`: [T5_SEQ, T5_DIM] — text context from T5 encoder (BF16)
- `clip_raw`: [CLIP_PAD, CLIP_DIM] — image context from CLIP encoder (BF16, padded)
- `ref_col`: [REF_N, REF_CONV_COL] — im2col'd reference latent (BF16)
- `timestep`: [1] — current diffusion timestep
- `weights`: DiT weights (time embedding, projections, transformer blocks, head)

**输出**:
- `noise_prediction`: [X_N, DIT_IN_DIM * 4] — predicted noise (or denoised latent)

**Pipeline 位置**:
- 核心生成步骤，执行多步去噪 (typically 50 steps)
- 每步: 预测噪声 → 更新 latent → 下一步
- 支持 CFG (Classifier-Free Guidance): 分别用 positive/negative prompt 推理，加权组合

**Golden Case 配置** (config.py):
- DIT_DIM = 128, DIT_HEADS = 4, DIT_LAYERS = 1

---

### 5. VAE Decoder (`vae_decoder.py`)

**职责**: 将潜在表示解码回视频帧，生成最终输出。

**架构**:
```
CausalConv3d(16→1536) →
Middle: ResBlock(1536) + Attention(1536) + ResBlock(1536)
Stage 0: ResBlock(1536→768) + ResBlock(768) + Upsample3d(t_up=T) → [1, 768, 2, 16, 16]
Stage 1: ResBlock(768→384) + ResBlock(384) + Upsample3d(t_up=T) → [1, 384, 4, 32, 32]
Stage 2: ResBlock(384→192) + ResBlock(192) + Upsample3d(t_up=F) → [1, 192, 4, 64, 64]
Stage 3: ResBlock(192→96) + ResBlock(96)
CausalConv3d(96→3) → [1, 3, 4, 64, 64]
→ clamp(-1, 1)
```

**输入**:
- `latents`: [1, 16, LAT_F, LAT_H, LAT_W] — denoised latent from DiT
- `weights`: VAE decoder weights

**输出**:
- `video`: [1, 3, T, H, W] — decoded video frames

**Pipeline 位置**:
- 将 DiT 生成的 denoised latent 解码为视频帧 tensor (值域 [-1, 1])
- WanVideoPipeline 内部对输出做 denormalize 并转为 PIL Images 后返回

**Golden Case 配置** (config.py):
- VAE_Z_DIM = 16, VAE_DEC_CH = [1536, 768, 384, 192, 96]
- LAT_F = 1 (encoder natural output), H = W = 64

---

## 公共配置 (`config.py`)

所有子网络共享的模型配置，包含:
- **Golden Case**: 小规模配置，用于快速数值验证
- **Real Case**: 完整规模配置，匹配真实模型 (注释状态，按需启用)

主要配置项:
- T5: T5_DIM, T5_SEQ, T5_LAYERS
- CLIP: CLIP_DIM, CLIP_TOKENS, CLIP_LAYERS
- VAE: VAE_Z_DIM, VAE_ENC_CH, VAE_DEC_CH
- DiT: DIT_DIM, DIT_HEADS, DIT_LAYERS
- Video: H, W, CHUNK_SIZE

---

## 依赖关系

```
vae_decoder.py ──→ vae_encoder.py (import vae_encoder as ve)
                    ├── 复用 NPU kernel runners (_compile, _run)
                    ├── 复用 building blocks (causal_conv3d_npu_fp32, resblock_npu, etc.)
                    ├── 复用 host helpers (_pad_cols, _to_2d, etc.)
                    └── 复用 compile cache (_cache)

其他子网络相互独立
```

**注意**: `vae_decoder.py` 依赖 `vae_encoder.py`，无法独立运行。如需独立部署，需将共享代码提取为公共模块。