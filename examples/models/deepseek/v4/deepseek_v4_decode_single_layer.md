# DeepSeek-V4 Single-Layer Decode Flow

One full `Block.forward` pass (model.py:689-701), single card, decode step
(S=1). Tensor shapes use real model dimensions: B=batch, T=B×1, D=7168,
H=128, HEAD_DIM=512, ROPE_DIM=64, Q_LORA=1536, HC=4.

Legend:
- `[orch]`    — orchestrator-only operation (no separate pypto kernel)
- `[EP-orch]` — requires inter-card AllToAllv; host orchestrator calls HCCL

---

## Top-level Block flow

```
═══════════════════════════════════════════════════════════════════════════════
  ENTRY: x  [B, 1, HC=4, D=7168]  bf16
         input_ids  [B, 1]  int64
═══════════════════════════════════════════════════════════════════════════════
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  attention.py                             ║
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
              ║  (hc_pre ffn + ffn_norm fused + gate)     ║
              ║                                           ║
              ║  IN : x [B,1,4,D]  bf16                   ║
              ║  OUT: indices [T, TOPK=6]  int32          ║
              ║       weights [T, TOPK=6]  fp32           ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [EP-orch]  dispatch                      │
              │  pack tokens by dest expert rank          │
              │  AllToAllv → recv_x, recv_expert_id,      │
              │              recv_weights                 │
              └───────────────────────────────────────────┘
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe_expert.py                            ║
              ║  model.py:636-644                         ║
              ║  (local routed + shared)                  ║
              ║                                           ║
              ║  IN : recv_x  [RECV_TOTAL, D]             ║
              ║       recv_expert_id, recv_weights        ║
              ║       x_local [T, D]  (for shared only)   ║
              ║       expert_w1/w2/w3 [N_LOCAL_EXPERTS,…] ║
              ║       shared_w1/w2/w3                     ║
              ║  OUT: recv_y [RECV_TOTAL, D]  bf16        ║
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
              ║       post_ffn [B,1,4], comb_ffn [B,1,4,4]║
              ║  OUT: x_next [B, 1, HC=4, D]  bf16        ║
              ╚═══════════════════════════════════════════╝
                              │
═══════════════════════════════════════════════════════════════════════════════
  EXIT: x_next [B, 1, HC=4, D=7168]  bf16
        → next Block (×61) → MTPBlock → ParallelHead → logits
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
║  NOTE: line 506 act_quant (FP8 QAT sim) is skipped on A3.                   ║
║                                                                             ║
║  IN :  x [B, S, D]                    bf16  (hc_pre output)                 ║
║        norm_w [D]                     fp32  (attn_norm gamma, fused)        ║
║        wq_a [D, Q_LORA=1536]          bf16                                  ║
║        wq_b [Q_LORA, H*HEAD_DIM]      bf16                                  ║
║        wkv  [D, HEAD_DIM=512]         bf16                                  ║
║        rope_cos/sin [T, ROPE_DIM=64]  bf16                                  ║
║        gamma_cq [Q_LORA]              bf16                                  ║
║        gamma_ckv [HEAD_DIM]           bf16                                  ║
║  OUT:  q   [T, H=128, HEAD_DIM=512]   bf16  (RoPE applied)                  ║
║        kv  [T, HEAD_DIM=512]          bf16  (RoPE applied)                  ║
║        qr  [T, Q_LORA=1536]           bf16  (reused by indexer)             ║
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
║  sparse_attn.py  (always called; ratio ∈ {0, 4, 128})                       ║
║  model.py:533-534                                                           ║
║                                                                             ║
║  IN : q [T,H,HEAD_DIM]                                                      ║
║       ori_kv (PA)              — always                                     ║
║       cmp_kv (PA)              — ratio>0 only                               ║
║       topk_idxs                — ratio-dependent, see § below               ║
║       attn_sink [H]  fp32                                                   ║
║       seqused_kv [B]                                                        ║
║       freqs_cos/sin                                                         ║
║  OUT: o [T, H, HEAD_DIM]  bf16                                              ║
║       (line 534 inverse RoPE fused)                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
         │ o
         │
         ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  o_proj.py  (grouped output LoRA)                                           ║
║  model.py:537-542                                                           ║
║                                                                             ║
║  IN :  o    [T, H=128, HEAD_DIM=512]              bf16                      ║
║        wo_a [O_GROUPS=16, O_LORA=1024, 4096]      bf16                      ║
║        wo_b [D=7168, O_GROUPS*O_LORA=16384]        bf16                     ║
║  OUT:  attn_out  [T, D=7168]  bf16                                          ║
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
║  compressor.py  (Attention.compressor, decode part)                ║
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
║  rotate=False, BF16 path                                           ║
╚════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║  indexer.py  (Indexer.forward, decode part)                        ║
║  model.py:511 (call), 402-433 (impl)                               ║
║                                                                    ║
║  IN    : x [B,1,D], qr [T,Q_LORA], start_pos, offset               ║
║          idx_wq_b, weights_proj, cos/sin, hadamard_q               ║
║  InOut : indexer's own kv_cache (PA pool, separate from cmp_kv)    ║
║          internal Compressor's kv_state/score_state                ║
║  OUT   : indexer_topk [T, IDX_TOPK=1024]  int32                    ║
║                                                                    ║
║  Internal: instantiates its own Compressor (rotate=True, FP4),     ║
║  distinct from Attention.compressor — different weights, different ║
║  KV pool, called inside indexer.py at model.py:417.                ║
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
                                                │ sparse_attn  │──► o
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
                                              ori_kv ⧺ cmp_kv │ sparse_attn │──► o
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
                                              ori_kv ⧺ cmp_kv │ sparse_attn │──► o
                                                              └──────────────┘
```

`window_topk_idxs` (line 507) and `get_compress_topk_idxs` (line 513) are pure
index computations from `(win, ratio, bsz, seqlen, start_pos)`; they live
inline in the orch and are NOT separate kernels.

---

## Draft file coverage

| Step | model.py | Draft file | Status |
|---|---|---|---|
| hc_pre (attn) | 691 | `hc_pre.py` | skeleton |
| qkv_proj_rope (attn_norm fused + Q/KV LoRA + RoPE) | 692, 495-504 | `qkv_proj_rope.py` | skeleton |
| act_quant (FP8 QAT sim) | 506 | — | skipped on A3 |
| kv → ori_kv write | 530 | [orch] scatter | — |
| indexer (decode) | 511 (call), 402-433 (impl) | `indexer.py` | skeleton |
| compressor (decode) | 532 (call), 316-377 (impl) | `compressor.py` | skeleton |
| window_topk_idxs | 507 | [orch] inline in sparse_attn caller | — |
| get_compress_topk_idxs (HCA) | 513 | [orch] inline in sparse_attn caller | — |
| topk_idxs concat | 514-515 | [orch] | — |
| sparse_attn (+ inverse RoPE fused) | 533-534 | `sparse_attn.py` | skeleton |
| o_proj | 537-542 | `o_proj.py` | skeleton |
| hc_post (attn) | 694 | `hc_post.py` | skeleton |
| moe_router (hc_pre ffn + ffn_norm + gate) | 697-698, 564-584 | `moe_router.py` | skeleton |
| EP dispatch | — | [EP-orch] HCCL AllToAllv | — |
| moe_expert | 636-644 | `moe_expert.py` | skeleton |
| EP combine | — | [EP-orch] HCCL AllToAllv | — |
| routed_y + sh | 644-645 | [orch] elementwise add | — |
| hc_post (ffn) | 700 | `hc_post.py` | skeleton |

## Layer-type conditional (compress_ratio)

| compress_ratio | Compressor | Indexer | sparse_attn topk source |
|---|---|---|---|
| 0   | not called      | not called | window_topk_idxs |
| 4   | called (ratio=4)   | called  | window_topk_idxs ⧺ indexer_topk |
| 128 | called (ratio=128) | not called | window_topk_idxs ⧺ get_compress_topk_idxs (HCA) |

## EP topology notes

- **moe_router**: runs on every card with replicated `gate_w`; indices cover
  global expert space `[0, N_EXPERTS=384)`.
- **moe_expert**: each card holds `N_LOCAL_EXPERTS = N_EXPERTS / EP_WORLD_SIZE`
  routed expert weights. `recv_x` is the post-dispatch token set (source:
  all cards); `x_local` is the original pre-dispatch local token set (shared
  expert only). The two inputs are distinct token populations.
- **shared expert**: computed locally on `x_local` with no communication;
  result `sh` stays on the card and is added after combine.
- **hc_post** (both attn and ffn): same draft file, called twice per Block
  with different post/comb tensors from the respective hc_pre call.
