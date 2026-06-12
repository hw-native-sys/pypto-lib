# ══════════════════════════════════════════════════════════════════════════════
# Fun-Control 1.3B — pypto3.0 model configuration.
#
# Two cases are tracked:
#   • Golden case — small-scale config for fast numerical validation against
#     test_golden_fun_control_full.py.  Active by default.
#   • Real case — full-size config matching infer_fun_control_1_3b_text.py
#     with the real open-clip-xlm-roberta-large-vit-huge-14 checkpoint.
#     Commented out; activate when ready.
# ══════════════════════════════════════════════════════════════════════════════

# ── Video / image ──
H, W = 64, 64
NUM_FRAMES = 5
CHUNK_SIZE = 5
FALLBACK_PROMPT = "robot arm manipulation"


# ══════════════════════════════════════════════════════════════════════════════
# T5 text encoder
# ══════════════════════════════════════════════════════════════════════════════

# ── Golden case (test_golden_fun_control_full.py:65-71) ──
T5_VOCAB = 1000
T5_DIM = 128
T5_FFN = 256
T5_HEADS = 4
T5_HEAD_DIM = 32
T5_LAYERS = 1
T5_SEQ = 32
T5_NUM_BUCKETS = 32

# ── Real case (DiffSynth-Studio t5_umt5-xxl-enc-bf16.pth) ──
# T5_VOCAB = 256384
# T5_DIM = 4096
# T5_FFN = 10240
# T5_HEADS = 64
# T5_HEAD_DIM = 64
# T5_LAYERS = 24
# T5_SEQ = 512  # Reduced from 512 for testing
# T5_NUM_BUCKETS = 32


# ══════════════════════════════════════════════════════════════════════════════
# CLIP image encoder
#
# Architecture: PatchConv2d → CLS+PosEmbed → PreLN →
#               [LN → MHA → Proj → Res → LN → FFN → Res] × L
#
# Source (golden): test_golden_fun_control_full.py:73-79, 127-142, 721-744
# Source (real):   DiffSynth-Studio/diffsynth/models/wan_video_image_encoder.py
#                  clip_xlm_roberta_vit_h_14()  → lines 822-849
#                  VisionTransformer.__init__    → lines 386-454
#                  VisionTransformer.forward     → lines 456-478
#                  AttentionBlock.__init__       → lines 289-330
#                  XLMRobertaCLIP.__init__       → lines 642-710
#                  WanImageEncoder.encode_image  → lines 864-878
# ══════════════════════════════════════════════════════════════════════════════

# ── Golden case (test_golden_fun_control_full.py:73-79) ──
#
# Reduced dimensions for fast numerical validation.
# Activation: QuickGELU  →  ff * sigmoid(1.702 * ff)     (golden:742)
# Layers run: all (CLIP_LAYERS = 2)
# Patch conv: Conv2d(3, 128, k=14, s=14, bias=False)     (no bias weight in golden:128)
# Output:     [1, 5, 128]  (full sequence including CLS)
#
CLIP_DIM = 128
CLIP_HEADS = 4
CLIP_HEAD_DIM = 32                # CLIP_DIM // CLIP_HEADS
CLIP_LAYERS = 2
CLIP_PATCH = 14
CLIP_IMG = 28                     # must be divisible by CLIP_PATCH
CLIP_TOKENS = 5                   # (CLIP_IMG // CLIP_PATCH) ** 2 + 1 = 4 + 1
CLIP_FFN = 512                    # CLIP_DIM * 4
CLIP_NORM_EPS = 1e-5

# ── Real case (open-clip-xlm-roberta-large-vit-huge-14) ──
#
# Full ViT-H/14 from OpenCLIP, loaded via:
#   ModelConfig(path="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
#
# Key differences from golden case:
#   1. Scale:     1280-dim, 16 heads, 32 layers (vs 128/4/2)
#   2. Image:     224×224 → 256 patches → 257 tokens (vs 28×28 → 4 → 5)
#   3. Activation: standard GELU (nn.GELU) — NOT QuickGELU
#                  Trace: clip_xlm_roberta_vit_h_14 sets activation='gelu' (line 835)
#                  → XLMRobertaCLIP(activation='gelu') (line 655)
#                  → VisionTransformer(activation='gelu') (line 702)
#                  → AttentionBlock(activation='gelu') (line 442)
#                  → nn.GELU() (line 320: else branch)
#                  Golden test uses QuickGELU (golden:742) — ACTIVATION MISMATCH
#   4. Blocks:    encode_image calls visual(x, use_31_block=True) (line 877)
#                  → runs transformer[:-1] = first 31 of 32 blocks (line 474)
#                  → early return, skips block 32 + post_norm + head (line 475)
#   5. Patch conv: Conv2d(3, 1280, k=14, s=14, bias=False)
#                  bias=False because pre_norm=True (line 429: bias=not pre_norm)
#   6. CLS token:  shape [1, 1, 1280], gain-init: 1/sqrt(1280) * randn
#   7. Pos embed:  shape [1, 257, 1280], gain-init: 1/sqrt(1280) * randn
#   8. Pre-LN:     LayerNorm(1280, eps=1e-5) — exists because pre_norm=True
#   9. Post-LN:    LayerNorm(1280, eps=1e-5) — created but SKIPPED (use_31_block)
#  10. Head:       nn.Parameter(1280, 1024) — created but SKIPPED (use_31_block)
#  11. Output:     [1, 257, 1280] raw hidden states after 31 blocks
#
# Preprocessing (WanImageEncoder.encode_image, lines 864-878):
#   PIL Image → resize (width, height) → preprocess_image: [-1, 1]
#   → bicubic resize to 224×224 → mul_(0.5).add_(0.5): [0, 1]
#   → Normalize(mean, std):
#       mean = [0.48145466, 0.4578275, 0.40821073]  (line 785)
#       std  = [0.26862954, 0.26130258, 0.27577711]  (line 786)
#
# Downstream usage (wan_video_dit.py WanModel.forward):
#   clip_feature [1, 257, 1280]
#   → dit.img_emb(clip_feature): MLP(1280 → dim)
#     = LayerNorm(1280) → Linear(1280,1280) → GELU → Linear(1280,dim) → LayerNorm(dim)
#   → torch.cat([clip_embedding, text_context], dim=1)
#
# CLIP_DIM = 1280
# CLIP_HEADS = 16
# CLIP_HEAD_DIM = 80              # 1280 // 16
# CLIP_LAYERS = 31                # use_31_block=True: first 31 of 32
# CLIP_PATCH = 14
# CLIP_IMG = 224
# CLIP_TOKENS = 257               # (224 // 14) ** 2 + 1 = 256 + 1
# CLIP_FFN = 5120                 # 1280 * 4
# CLIP_NORM_EPS = 1e-5
# CLIP_ACTIVATION = "gelu"        # standard GELU, NOT QuickGELU
# CLIP_EMBED_DIM = 1024           # joint text-vision embedding (unused by vision path)
# CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
# CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ══════════════════════════════════════════════════════════════════════════════
# VAE encoder / decoder
# ══════════════════════════════════════════════════════════════════════════════

VAE_Z_DIM = 16
VAE_SPATIAL_FACTOR = 8
VAE_TEMPORAL_FACTOR = 4
VAE_ENC_CH = [96, 192, 384, 768, 1536]
VAE_DEC_CH = [1536, 768, 384, 192, 96]


# ══════════════════════════════════════════════════════════════════════════════
# DiT denoising network
# ══════════════════════════════════════════════════════════════════════════════

DIT_DIM = 128
DIT_HEADS = 4
DIT_HEAD_DIM = 32
DIT_FFN = 256
DIT_LAYERS = 1
DIT_IN_DIM = 16
DIT_TEXT_DIM = 128
DIT_FREQ_DIM = 64
DIT_EPS = 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

NUM_STEPS = 1
SHIFT = 5.0
CFG_SCALE = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Latent space (derived)
# ══════════════════════════════════════════════════════════════════════════════

LAT_F = 2                         # (CHUNK_SIZE - 1) // VAE_TEMPORAL_FACTOR + 1
LAT_H = 8                         # H // VAE_SPATIAL_FACTOR
LAT_W = 8                         # W // VAE_SPATIAL_FACTOR