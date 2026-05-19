# DeepSeek-V4 Single-Layer Decode Flow

One full `Block.forward` pass (model.py:689-701), single card, decode step.
Tensor shapes use real model dimensions for **DeepSeek-V4-Flash**:
B=batch, S=2, T=B×S=2B, D=4096, H=64, HEAD_DIM=512, ROPE_DIM=64,
Q_LORA=1024, HC=4, IDX_TOPK=512, N_EXPERTS=16 (local, per EP card),
TOPK=6.

Pro values differ: D=7168, H=128, Q_LORA=1536, O_GROUPS=16, IDX_TOPK=1024,
N_EXPERTS=384, N_BLOCKS=61.

Legend:
- `[orch]`    — orchestrator-only operation (no separate pypto kernel)
- `[EP-orch]` — requires inter-card AllToAllv; host orchestrator calls HCCL.
  Current `moe.py` runs single-card (`EP_WORLD_SIZE=1`); the multi-card
  exchange is not yet wired (see [EP topology notes](#ep-topology-notes)).

---

## Top-level Block flow

`Block.forward` reduces to **attention → moe**, where each macro-kernel
already wraps its own `hc_pre` / `hc_post` pair around the inner compute.
Per-ratio dispatch happens at the orchestration level: one of
`decode_swa` / `decode_csa` / `decode_hca` is compiled per layer
(`compress_ratio == 0 / 4 / 128`) and calls the matching
`attention_{swa,csa,hca}` followed by the shared `moe`.

```
═══════════════════════════════════════════════════════════════════════════════
  ENTRY: x  [B, S, HC=4, D=4096]  bf16
         input_ids  [B, S]  int64
═══════════════════════════════════════════════════════════════════════════════
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  attention_{swa,csa,hca}.py               ║
              ║  model.py:691-694                         ║
              ║  (hc_pre[attn] + qkv_proj_rope            ║
              ║   + [compressor] + [indexer]              ║
              ║   + sparse_attn + hc_post[attn])          ║
              ║                                           ║
              ║  IN : x [B,S,HC=4,D]  bf16              ║
              ║  OUT: x [B,S,HC=4,D]  bf16              ║
              ║                                           ║
              ║  See "ATTENTION breakdown" below.         ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe.py                                   ║
              ║  model.py:696-700                         ║
              ║  (hc_pre[ffn] + moe_router                ║
              ║   + moe_dispatch + moe_expert             ║
              ║   + moe_combine + hc_post[ffn])           ║
              ║                                           ║
              ║  IN : x [B,S,HC=4,D]  bf16              ║
              ║       input_ids [B,S] int64             ║
              ║       (router/expert weights …)           ║
              ║  OUT: x_next [B,S,HC=4,D]  bf16         ║
              ║                                           ║
              ║  See "MoE breakdown" below.               ║
              ╚═══════════════════════════════════════════╝
                              │
═══════════════════════════════════════════════════════════════════════════════
  EXIT: x_next [B, S, HC=4, D=4096]  bf16
        → next Block (×43 for Flash) → MTPBlock → ParallelHead → logits
═══════════════════════════════════════════════════════════════════════════════
```

The `decode_{swa,csa,hca}` orchestrators in this directory wire these two
macro-kernels together for a single layer; the rest of this document
walks into each macro-kernel.

---

## ATTENTION breakdown

Corresponds to `Block.hc_pre(attn)` + `self.attn_norm` + `Attention.forward`
+ `Block.hc_post(attn)`, model.py:691-694. Each
`attention_{swa,csa,hca}.py` is a `@pl.jit.inline` composition of the
sub-kernels below; the variant determines whether `compressor` and
`indexer` participate.

```
  IN: x [B, S, HC=4, D]  bf16
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_pre.py  (attn)                                                          ║
║  model.py:691                                                               ║
║                                                                             ║
║  IN :  x          [B, S, HC=4, D]  bf16                                   ║
║        hc_attn_fn [24, HC*D]         fp32                                   ║
║        hc_attn_scale [3]             fp32                                   ║
║        hc_attn_base  [24]            fp32                                   ║
║  OUT:  x_mixed   [B, S, D]         bf16  ← 4 copies merged into 1         ║
║        post_attn [B, S, 4]         fp32  ← saved for hc_post              ║
║        comb_attn [B, S, 4, 4]      fp32  ← saved for hc_post              ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ x_mixed [B,S,D]
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  qkv_proj_rope.py  (attn_norm fused + Q/KV LoRA + RoPE)                     ║
║  model.py:692, 495-504                                                      ║
║  NOTE: W8A8C16: kv stays BF16 (attn KV Cache C16).                          ║
║        flash: act_quant on kv non-rope dims (L506, KV cache C8 sim).        ║
║        rope_cos/sin shape [T, ROPE_DIM] carries 2 consecutive positions     ║
║        per batch (start_pos and start_pos+1).                 ║
║                                                                             ║
║  IN :  x [B, S, D]                  bf16  (hc_pre output)                 ║
║        norm_w [D]                     fp32  (attn_norm gamma, fused)        ║
║        wq_a [D, Q_LORA=1024]          bf16                                  ║
║        wq_b [Q_LORA, H*HEAD_DIM]      bf16                                  ║
║        wkv  [D, HEAD_DIM=512]         bf16                                  ║
║        rope_cos/sin [T, ROPE_DIM=64] bf16                                ║
║        gamma_cq [Q_LORA]              bf16                                  ║
║        gamma_ckv [HEAD_DIM]           bf16                                  ║
║  OUT:  q   [T, H=64, HEAD_DIM=512] bf16  (RoPE applied per token)        ║
║        kv  [T, HEAD_DIM=512]       bf16  (RoPE applied per token)        ║
║        qr  [T, Q_LORA=1024]        bf16  (reused by indexer)             ║
╚═════════════════════════════════════════════════════════════════════════════╝
         │ q               │ kv                   │ qr
         │                 │                      │
         │     kv → write ori_kv cache  [orch]    │
         │     for s in 0..S-1:                   │
         │       ori_kv[block, (start_pos+s)%WIN] │
         │           = kv[b*S+s, :]               │
         │     model.py:530                       │
         │                 │                      │
         │             ori_kv (PA)                │
         │                 │                      │
         │  ┌──── cmp_kv (PA, ratio>0 only) ──────┤
         │  │   cmp scatter fires once per call   │
         │  │   when (start_pos+S)%ratio == 0     │
         │  │                                     │
         │  │   topk_idxs [T, *] — per-token   │
         │  │   (indexer/HCA produces 2 rows per  │
         │  │   batch; see § below) ──────────────┤
         ▼  ▼                                     ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  sparse_attn.py  (always called; ratio ∈ {0, 4, 128}; o_proj fused)         ║
║  model.py:533-534, 537-542                                                  ║
║  NOTE: outer loop is `for t in pl.range(T)` — per-query-token attention. ║
║        Token t belongs to batch b = t // S and step s = t % S; each token   ║
║        reads seqused_kv[b, s]. Intra-query causal is enforced upstream by   ║
║        the topk index set the indexer produces per token.                  ║
║                                                                             ║
║  IN : q [T, H, HEAD_DIM]                                                 ║
║       ori_kv (PA)              — always                                     ║
║       cmp_kv (PA)              — ratio>0 only                               ║
║       topk_idxs [T, *]         — per-token; ratio-dependent, see § below    ║
║       attn_sink [H]  fp32                                                   ║
║       seqused_kv [B, S]        — per-token valid sparse KV length           ║
║       freqs_cos/sin [T, ROPE_DIM]                                           ║
║       wo_a [O_GROUPS=8, O_LORA=1024, 4096]   bf16   (grouped output LoRA)   ║
║       wo_b [D=4096, O_GROUPS*O_LORA=8192]    int8                           ║
║       wo_b_scale [D]                         fp32                           ║
║  OUT: attn_out [T, D=4096]  bf16                                         ║
║       (line 534 inverse RoPE + line 537-542 o_proj fused)                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ attn_out [T, D]
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_post.py  (attn)                                                         ║
║  model.py:694                                                               ║
║                                                                             ║
║  IN :  x        [B, S, D]        bf16  (attn_out)                         ║
║        residual [B, S, HC=4, D]  bf16                                     ║
║        post     [B, S, 4]        fp32                                     ║
║        comb     [B, S, 4, 4]     fp32                                     ║
║  OUT:  y  [B, S, HC=4, D]        bf16                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │
  OUT: x [B, S, HC=4, D]  bf16  → top-level moe.py
```

---

## MoE breakdown

Corresponds to `Block.hc_pre(ffn)` + `self.ffn_norm` + `MoE.forward` +
`Block.hc_post(ffn)`, model.py:696-700. `moe.py` is the
`@pl.jit.inline` composition; the layout below mirrors its six stages.

```
  IN: x_hc [B, S, HC=4, D]  bf16   (attention output)
      input_ids [B, S]      int64
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_pre.py  (ffn)                                                           ║
║  model.py:697                                                               ║
║                                                                             ║
║  IN :  x_hc        [B, S, HC=4, D]  bf16                                  ║
║        hc_ffn_fn   [24, HC*D]         fp32                                  ║
║        hc_ffn_scale [3]               fp32                                  ║
║        hc_ffn_base  [24]              fp32                                  ║
║  OUT:  x_mixed    [B, S, D]         bf16  ← 4 copies merged into 1        ║
║        post_ffn   [B, S, 4]         fp32  ← saved for hc_post(ffn)        ║
║        comb_ffn   [B, S, 4, 4]      fp32  ← saved for hc_post(ffn)        ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ x_mixed [B,S,D]
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  moe_router.py  (ffn_norm fused + gate + topk + hash route)                 ║
║  model.py:564-584                                                           ║
║                                                                             ║
║  IN :  x_mixed     [B, S, D]        bf16                                  ║
║        norm_w      [D]                fp32  (ffn_norm gamma, fused)         ║
║        gate_w      [N_EXPERTS, D]     fp32                                  ║
║        gate_bias   [N_EXPERTS]        fp32                                  ║
║        tid2eid     [VOCAB, TOPK]      int32  (hash-routed layers)           ║
║        input_ids   [B, S]           int64                                 ║
║        layer_id    scalar             int32                                 ║
║  OUT:  x_norm      [T, D]          bf16  (post ffn_norm hidden state)    ║
║        indices     [T, TOPK=6]     int32                                 ║
║        weights     [T, TOPK=6]     fp32                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ x_norm, indices, weights
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  moe_dispatch.py  (per-token INT8 quant + pack by destination expert)       ║
║  [EP-orch placeholder — currently single-card, EP_WORLD_SIZE=1]             ║
║  NOTE: RECV_MAX = B*S*TOPK under MTP (was 384 at S=1).            ║
║                                                                             ║
║  IN :  x_norm, indices, weights                                             ║
║  OUT:  recv_x            [N_LOCAL_EXPERTS, RECV_MAX, D]  int8           ║
║        recv_scale_dq     [N_LOCAL_EXPERTS, RECV_MAX]     fp32  (per-token)  ║
║        recv_weights      [N_LOCAL_EXPERTS, RECV_MAX]     fp32               ║
║        recv_token        [N_LOCAL_EXPERTS, RECV_MAX]     int32              ║
║        recv_expert_count [N_LOCAL_EXPERTS, 1]            int32              ║
║                                                                             ║
║  Pack loop (logical):                                                       ║
║    for p = t*TOPK + k:   # t walks T = B*S = 2B tokens                      ║
║      e = indices[p];  slot = recv_expert_count[e];                          ║
║      recv_x[e,slot,:] = INT8(x_norm[t,:]); recv_scale_dq[e,slot] = scale[t] ║
║      recv_weights[e,slot] = weights[p]   ; recv_token[e,slot] = t           ║
║      recv_expert_count[e] += 1                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  moe_expert.py  (routed expert GEMMs + shared expert; W8A8)                 ║
║  model.py:636-644                                                           ║
║                                                                             ║
║  IN :  recv_x            [N_LOCAL_EXPERTS, RECV_MAX, D]  int8           ║
║        recv_scale_dq     [N_LOCAL_EXPERTS, RECV_MAX]     fp32               ║
║        recv_expert_count_full [N_LOCAL_EXPERTS, 1]       int32  (= RECV_MAX,║
║                                                                  static)    ║
║        x_local           [T, D]   bf16  (= x_norm; for shared expert)    ║
║        expert_w1/w2/w3   [N_LOCAL_EXPERTS, …]  int8 + fp32 scale            ║
║        shared_w1/w2/w3   [...]                 int8 + fp32 scale            ║
║  OUT:  recv_y            [N_LOCAL_EXPERTS, RECV_MAX, D]  bf16           ║
║        sh                [T, D]                       bf16  (shared)     ║
║                                                                             ║
║  NOTE: expert loop walks the static count; tail rows beyond the actual      ║
║        recv_expert_count are still produced but contribute weight 0 in     ║
║        combine, so they are dropped.                                        ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ recv_y, sh
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  moe_combine.py  (weighted scatter-add back to token space + shared add)    ║
║  [EP-orch placeholder — currently single-card]                              ║
║                                                                             ║
║  IN :  recv_y, recv_token, recv_weights, recv_expert_count, sh              ║
║  OUT:  ffn_out [B, S, D]  bf16                                            ║
║                                                                             ║
║  Reduction (logical):                                                       ║
║    routed_y[t,:] = Σ over valid (e,slot) where recv_token[e,slot]==t        ║
║                    of recv_weights[e,slot] * recv_y[e,slot,:]               ║
║    ffn_out[t,:]  = routed_y[t,:] + sh[t,:]   # t walks T = 2B tokens        ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ ffn_out [B,S,D]
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_post.py  (ffn)                                                          ║
║  model.py:700                                                               ║
║                                                                             ║
║  IN :  ffn_out  [B, S, D]        bf16                                     ║
║        residual [B, S, HC=4, D]  bf16  (= moe.py input x_hc)              ║
║        post_ffn [B, S, 4]        fp32                                     ║
║        comb_ffn [B, S, 4, 4]     fp32                                     ║
║  OUT:  x_next   [B, S, HC=4, D]  bf16                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │
  OUT: x_next [B, S, HC=4, D]  bf16  → next Block
```

### EP topology notes

Today `moe.py` runs **single-card** (`EP_WORLD_SIZE=1`, `N_LOCAL_EXPERTS == N_EXPERTS`).
`moe_dispatch` and `moe_combine` are written as in-kernel pack/scatter so
the end-to-end MoE composes inline without HCCL. They are marked
`[EP-orch placeholder]` because the multi-card variant still needs
`AllToAllv` between dispatch and expert and again before combine.

**Reference implementation** for the multi-card path lives in
`simpler/examples/workers/l3/ep_dispatch_combine/` and already runs
end-to-end dispatch + combine on 2-card hardware via a single
orchestration kernel + three AIV children over a shared HCCL window
scratch (`dispatch.cpp`, `local_expert.cpp`, `combine.cpp`,
`ep_dispatch_combine_orch.cpp`). pypto has not yet adapted that path;
once the equivalent DSL primitives (TPUT/TNOTIFY barriers + HCCL window)
are exposed, `moe_dispatch` / `moe_combine` can be split into the real
pre/post-AllToAllv halves following the simpler reference.

EP semantics around the MoE sub-kernels (when EP > 1):

- **moe_router**: runs on every card with replicated `gate_w`; `indices`
  cover global expert space `[0, N_EXPERTS_GLOBAL)`. `x_norm` is the source
  for both `recv_x` (dispatch) and `x_local` (shared expert input).
- **moe_expert**: each card holds `N_LOCAL_EXPERTS = N_EXPERTS_GLOBAL / EP_WORLD_SIZE`
  routed expert weights. `recv_x` is the post-dispatch token set (source:
  all cards' `x_norm`, repacked by destination expert); `x_local` is this
  card's slice of `x_norm` (shared expert only). The two inputs are
  distinct token populations from the same global `x_norm`.
- **shared expert**: computed locally on `x_local` with no communication;
  result `sh` stays on the card and is added inside `moe_combine`.
