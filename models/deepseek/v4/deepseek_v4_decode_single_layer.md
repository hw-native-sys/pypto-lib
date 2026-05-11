# DeepSeek-V4 Single-Layer Decode Flow

One full `Block.forward` pass (model.py:689-701), single card, decode step
(S=1). Tensor shapes use real model dimensions for **DeepSeek-V4-Flash**:
B=batch, T=B×1, D=4096, H=64, HEAD_DIM=512, ROPE_DIM=64, Q_LORA=1024, HC=4.

Pro values differ: D=7168, H=128, Q_LORA=1536, O_GROUPS=16, IDX_TOPK=1024,
N_EXPERTS=384, N_BLOCKS=61.

Legend:
- `[orch]`    — orchestrator-only operation (no separate pypto kernel)
- `[EP-orch]` — requires inter-card AllToAllv; host orchestrator calls HCCL

---

## Top-level Block flow

```
═══════════════════════════════════════════════════════════════════════════════
  ENTRY: x  [B, 1, HC=4, D=4096]  bf16
         input_ids  [B, 1]  int64
═══════════════════════════════════════════════════════════════════════════════
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  attention_{swa,csa,hca}.py               ║
              ║  model.py:691-694  (see breakdown below)  ║
              ║                                           ║
              ║  IN : x [B,1,4,D]  bf16                   ║
              ║  OUT: x [B,1,4,D]  bf16                   ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe_router.py                            ║
              ║  model.py:697-698, 564-584                ║
              ║  (hc_pre ffn + ffn_norm + gate)           ║
              ║  hc_pre reuses hc_pre.py with hc_ffn_*    ║
              ║                                           ║
              ║  IN : x_hc [B,1,HC=4,D]  bf16             ║
              ║       hc_ffn_fn/scale/base                ║
              ║       norm_w, gate_w, gate_bias           ║
              ║       tid2eid, input_ids   (hash mode)    ║
              ║  OUT: x_norm   [T, D]            bf16     ║
              ║         → dispatch recv_x source          ║
              ║         → moe_expert x_local (shared)     ║
              ║       indices  [T, TOPK=6]       int32    ║
              ║       weights  [T, TOPK=6]       fp32     ║
              ║       post_ffn [B, 1, 4]         fp32     ║
              ║       comb_ffn [B, 1, 4, 4]      fp32     ║
              ║         → hc_post (ffn) post/comb         ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [EP-orch]  dispatch                      │
              │  inputs : x_norm, indices, weights        │
              │  pack tokens by dest expert rank          │
              │  AllToAllv → recv_x, recv_weights,        │
              │              recv_expert_count, recv_idx  │
              │  (per-expert 3D layout, see moe_expert)   │
              └───────────────────────────────────────────┘
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe_expert.py                            ║
              ║  model.py:636-644                         ║
              ║  (local routed + shared; W8A8 GEMM)       ║
              ║                                           ║
              ║  IN : recv_x [N_LOCAL_EXPERTS, RECV_MAX,D]║
              ║       recv_weights  [N_LOCAL_EXPERTS,…]   ║
              ║       recv_expert_count [N_LOCAL_EXPERTS] ║
              ║       x_local [T, D]  (= x_norm; shared)  ║
              ║       expert_w1/w2/w3 [N_LOCAL_EXPERTS,…] ║
              ║       shared_w1/w2/w3                     ║
              ║  OUT: recv_y [N_LOCAL_EXPERTS, RECV_MAX,D]║
              ║       sh     [T, D]           bf16        ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [EP-orch]  combine                       │
              │  AllToAllv → scatter_add recv_y per token │
              │  → routed_y [T, D]  bf16                  │
              └───────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [orch]  ffn_out = routed_y + sh          │
              │  model.py:644-645                         │
              └─────────────────────┬─────────────────────┘
                                    │
                                    ▼
              ╔═══════════════════════════════════════════╗
              ║  hc_post.py  (ffn)                        ║
              ║  model.py:700                             ║
              ║                                           ║
              ║  IN : ffn_out [B,1,D], residual [B,1,4,D] ║
              ║       (residual = moe_router input x_hc)  ║
              ║       post_ffn [B,1,4], comb_ffn [B,1,4,4]║
              ║  OUT: x_next [B, 1, HC=4, D]  bf16        ║
              ╚═══════════════════════════════════════════╝
                              │
═══════════════════════════════════════════════════════════════════════════════
  EXIT: x_next [B, 1, HC=4, D=4096]  bf16
        → next Block (×43) → MTPBlock → ParallelHead → logits
═══════════════════════════════════════════════════════════════════════════════
```

---

## ATTENTION breakdown

Corresponds to `Block.hc_pre` + `self.attn_norm` + `Attention.forward` +
`Block.hc_post`, model.py:691-694.

```
  IN: x [B, 1, HC=4, D]  bf16
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_pre.py  (attn)                                                          ║
║  model.py:691                                                               ║
║                                                                             ║
║  IN :  x          [B, 1, HC=4, D]    bf16                                   ║
║        hc_attn_fn [24, HC*D]         fp32                                   ║
║        hc_attn_scale [3]             fp32                                   ║
║        hc_attn_base  [24]            fp32                                   ║
║  OUT:  x_mixed   [B, 1, D]           bf16  ← 4 copies merged into 1         ║
║        post_attn [B, 1, 4]           fp32  ← saved for hc_post              ║
║        comb_attn [B, 1, 4, 4]        fp32  ← saved for hc_post              ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ x_mixed [B,1,D]
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  qkv_proj_rope.py  (attn_norm fused + Q/KV LoRA + RoPE)                     ║
║  model.py:692, 495-504                                                      ║
║  NOTE: W8A8C16: kv stays BF16 (attn KV Cache C16).                          ║
║        flash: act_quant on kv non-rope dims (L506, KV cache C8 sim).        ║
║                                                                             ║
║  IN :  x [B, S, D]                    bf16  (hc_pre output)                 ║
║        norm_w [D]                     fp32  (attn_norm gamma, fused)        ║
║        wq_a [D, Q_LORA=1024]          bf16                                  ║
║        wq_b [Q_LORA, H*HEAD_DIM]      bf16                                  ║
║        wkv  [D, HEAD_DIM=512]         bf16                                  ║
║        rope_cos/sin [T, ROPE_DIM=64]  bf16                                  ║
║        gamma_cq [Q_LORA]              bf16                                  ║
║        gamma_ckv [HEAD_DIM]           bf16                                  ║
║  OUT:  q   [T, H=64, HEAD_DIM=512]    bf16  (RoPE applied)                  ║
║        kv  [T, HEAD_DIM=512]          bf16  (RoPE applied)                  ║
║        qr  [T, Q_LORA=1024]           bf16  (reused by indexer)             ║
╚═════════════════════════════════════════════════════════════════════════════╝
         │ q               │ kv                   │ qr
         │                 │                      │
         │     kv → write ori_kv cache  [orch]    │
         │     ori_kv[block, slot % WIN] = kv     │
         │     model.py:530                       │
         │                 │                      │
         │             ori_kv (PA)                │
         │                 │                      │
         │  ┌──── cmp_kv (PA, ratio>0 only) ──────┤
         │  │                                     │
         │  │   topk_idxs (built by orch,         │
         │  │   ratio-dependent, see § below) ────┤
         ▼  ▼                                     ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  sparse_attn.py  (always called; ratio ∈ {0, 4, 128}; o_proj fused)         ║
║  model.py:533-534, 537-542                                                  ║
║                                                                             ║
║  IN : q [T,H,HEAD_DIM]                                                      ║
║       ori_kv (PA)              — always                                     ║
║       cmp_kv (PA)              — ratio>0 only                               ║
║       topk_idxs                — ratio-dependent, see § below               ║
║       attn_sink [H]  fp32                                                   ║
║       seqused_kv [B]                                                        ║
║       freqs_cos/sin                                                         ║
║       wo_a [O_GROUPS=8, O_LORA=1024, 4096]   bf16   (grouped output LoRA)   ║
║       wo_b [D=4096, O_GROUPS*O_LORA=8192]    int8                           ║
║       wo_b_scale [D]                         fp32                           ║
║  OUT: attn_out [T, D=4096]  bf16                                            ║
║       (line 534 inverse RoPE + line 537-542 o_proj fused)                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ attn_out [T, D]
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_post.py  (attn)                                                         ║
║  model.py:694                                                               ║
║                                                                             ║
║  IN :  x        [B, 1, D]          bf16  (attn_out)                         ║
║        residual [B, 1, HC=4, D]    bf16                                     ║
║        post     [B, 1, 4]          fp32                                     ║
║        comb     [B, 1, 4, 4]       fp32                                     ║
║  OUT:  y  [B, 1, HC=4, D]          bf16                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │
  OUT: x [B, 1, HC=4, D]  bf16  → top-level moe_router.py
```

---

## ratio-dependent inputs to sparse_attn

The main diagram treats `cmp_kv` and `topk_idxs` as black-box inputs. Their
construction depends on `compress_ratio`.

### Kernels involved (only when called)

```
╔════════════════════════════════════════════════════════════════════╗
║  compressor_ratio{4,128}.py  (Attention.compressor, decode part)   ║
║  model.py:532 (call), 316-377 (impl)                               ║
║                                                                    ║
║  IN    : x [B,1,D], start_pos                                      ║
║          wkv, wgate, ape, norm_w, cos/sin                          ║
║  InOut : kv_state, score_state          (persistent state buffers) ║
║          cmp_kv (PA pool slot)          (kernel-internal write,    ║
║                                          line 376, conditional on  ║
║                                          should_compress)          ║
║  OUT   : (return value discarded by decode caller)                 ║
║                                                                    ║
║  rotate=False (attn-mode).                                         ║
║  W8A8C16: not quantized (output BF16 to attn KV Cache C16).        ║
║  flash: act_quant on kv non-rope dims (KV cache C8 sim).           ║
╚════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║  indexer.py  (Indexer.forward, decode part)                        ║
║  model.py:511 (call), 402-433 (impl)                               ║
║                                                                    ║
║  IN    : x [B,1,D], qr [T,Q_LORA], start_pos, offset               ║
║          idx_wq_b, weights_proj, cos/sin, hadamard_q               ║
║  InOut : indexer's own kv_cache (PA pool, separate from cmp_kv)    ║
║          internal Compressor's kv_state/score_state                ║
║  OUT   : indexer_topk [T, IDX_TOPK=512]   int32                    ║
║                                                                    ║
║  Internal: instantiates its own Compressor (rotate=True),          ║
║  distinct from Attention.compressor — different weights, different ║
║  KV pool, called inside indexer.py at model.py:417.                ║
║  W8A8C16: A8 per-token-head int8 output (writes Indexer Cache C8). ║
║  flash: FP4 simulation (full Hadamard + fp4_act_quant).            ║
╚════════════════════════════════════════════════════════════════════╝
```

### Per-ratio wiring

```
ratio == 0  (SWA)
─────────────────────────────────────────────────────────
                 ┌────────────────┐
            x ──►│ qkv_proj_rope  │──► q ───────────────┐
                 └────────────────┘──► kv ─►[orch]──┐   │
                                  └─► qr (unused)   │   │
                                                    ▼   │
                                                ori_kv  │
              [orch] window_topk_idxs ──► topk_idxs ──► │
                                                        ▼
                                                ┌──────────────┐
                                                │ sparse_attn  │──► attn_out
                                                └──────────────┘


ratio == 4  (CSA)
─────────────────────────────────────────────────────────
                 ┌────────────────┐
            x ──►│ qkv_proj_rope  │──► q ─────────────────────────┐
            │    └────────────────┘──► kv ─►[orch] ori_kv ─────┐  │
            │                       └► qr ─┐                   │  │
            │                              │                   │  │
            ├───────────────────────►┌─────▼──────┐            │  │
            │                        │  indexer   │─►idx_topk─┐│  │
            │                        └────────────┘           ││  │
            │                                                 ││  │
            └─────────►┌──────────────┐                       ││  │
                       │  compressor  │──►(internal write)──► cmp_kv
                       └──────────────┘                       ││  │
                                                              ▼▼  ▼
        [orch] window_topk_idxs ⧺ idx_topk ──► topk_idxs ─►┌──────────────┐
                                              ori_kv ⧺ cmp_kv │ sparse_attn │──► attn_out
                                                            └──────────────┘


ratio == 128  (HCA)
─────────────────────────────────────────────────────────
                 ┌────────────────┐
            x ──►│ qkv_proj_rope  │──► q ─────────────────────────┐
            │    └────────────────┘──► kv ─►[orch] ori_kv ─────┐  │
            │                       └► qr (unused)             │  │
            │                                                  │  │
            └─────────►┌──────────────┐                        │  │
                       │  compressor  │──►(internal write)──► cmp_kv
                       └──────────────┘                        │  │
                                                               ▼  ▼
        [orch] window_topk_idxs ⧺ get_compress_topk_idxs ──► topk_idxs
                                              ori_kv ⧺ cmp_kv │ sparse_attn │──► attn_out
                                                              └──────────────┘
```

`window_topk_idxs` (line 507) and `get_compress_topk_idxs` (line 513) are pure
index computations from `(win, ratio, bsz, seqlen, start_pos)`; they live
inline in the orch and are NOT separate kernels.

---

## Layer-type conditional (compress_ratio)

| compress_ratio | Compressor | Indexer | sparse_attn topk source |
|---|---|---|---|
| 0   | not called      | not called | window_topk_idxs |
| 4   | called (ratio=4)   | called  | window_topk_idxs ⧺ indexer_topk |
| 128 | called (ratio=128) | not called | window_topk_idxs ⧺ get_compress_topk_idxs (HCA) |

## EP topology notes

**Reference implementation**: `simpler/examples/workers/l3/ep_dispatch_combine/`.
simpler already runs end-to-end dispatch + combine on 2-card hardware
via a single orchestration kernel + three AIV children over a shared
HCCL window scratch:

- `kernels/aiv/dispatch.cpp` — count exchange + 3-channel push
  (x BF16, weight FP32 1×W_PAD, idx INT32 1×IDX_PAD) + stage-out
  → `recv_x`, `recv_w`, `recv_idx`, `recv_count`.
- `kernels/aiv/local_expert.cpp` — placeholder for the production
  `moe_expert` kernel.
- `kernels/aiv/combine.cpp` — TPUT scatter `recv_y` by `recv_idx` into
  `routed_y_buf[t, k, :]` (relies on HCCL window zero-init), then
  reduce_sum along TOPK → `routed_y` FP32.
- `kernels/orchestration/ep_dispatch_combine_orch.cpp` — phase scheduler
  (histogram → publish → prefix_sum → payload_push → stage_out).

**pypto has not yet adapted** this path. The diagrams above mark these
stages `[EP-orch]` as a placeholder; once the equivalent DSL primitives
(TPUT/TNOTIFY barriers + HCCL window) are exposed in pypto, the
dispatch/combine boxes can be filled in following the simpler reference.

EP semantics around the MoE kernels:

- **moe_router**: runs on every card with replicated `gate_w`; indices cover
  global expert space `[0, N_EXPERTS=256)`. Also produces `x_norm` (post
  ffn_norm hidden state) which is the source for both dispatch's `recv_x`
  and moe_expert's `x_local`.
- **moe_expert**: each card holds `N_LOCAL_EXPERTS = N_EXPERTS / EP_WORLD_SIZE`
  routed expert weights. `recv_x` is the post-dispatch token set (source:
  all cards' x_norm, repacked by destination expert); `x_local` is this
  card's slice of x_norm (shared expert only). The two inputs are distinct
  token populations from the same global x_norm.
- **shared expert**: computed locally on `x_local` with no communication;
  result `sh` stays on the card and is added after combine.
