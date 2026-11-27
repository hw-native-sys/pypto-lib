# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gate `reference.recurrent_delta_rule` six ways: the chunked path, the prefill
hand-off, transformers' own recurrent function, the real module's decode path, the
real layer-0 weights, and `decode_delta_rule`'s host golden rolled out.

Host only; no device. The decode kernels have no PTO-ISA reference to read, so
this file is the oracle they are measured against, and it has to be right before
any of them is written. Gate F closes the other end: the device kernel is checked
against that golden on hardware, so tying the golden to the oracle here ties the
kernel to the oracle without a card.

    python models/qwen3_8_27b/test_decode_reference.py
"""
import argparse

import torch

import reference
from config import QWEN3_8_27B


# The recurrence is a Python loop over tokens and heads, so the gates run at a
# small head count first and at the model's real one second. Both keep
# `H // HG = 3`, the ratio the index mapping stands or falls on.
T = 64
H, HG = 6, 2
D = QWEN3_8_27B.linear_key_head_dim
CHUNK = 16

REAL_H = QWEN3_8_27B.linear_num_value_heads      # 48
REAL_HG = QWEN3_8_27B.linear_num_key_heads       # 16

# Float64 against float64 with a different summation order. Nothing here is
# allowed to be approximate: a real mismatch is a layout or scaling bug, not
# rounding.
EXACT_TOL = 1e-12


_GEOMETRY = (H, HG)


def _inputs(t: int = T, seed: int = 42) -> dict[str, torch.Tensor]:
    h, hg = _GEOMETRY
    return reference.make_inputs(t, h, D, hg, seed=seed)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def check_against_chunked() -> bool:
    """A: from a zero state the recurrent form must equal the chunked one.

    Same arithmetic, different grouping -- the chunked path batches a chunk's
    tokens through six stages, the recurrent path walks them one at a time. Any
    disagreement beyond float64 regrouping is a real difference in the maths.
    """
    st = _inputs()
    chunked = reference.delta_rule(st["q"], st["k"], st["v"], st["beta"], st["g"], CHUNK)
    o_rec, state = reference.recurrent_delta_rule(st["q"], st["k"], st["v"], st["beta"], st["g"])
    o_rel = _rel(o_rec, chunked["o"])
    s_rel = _rel(state, chunked["final_state"])
    ok = o_rel <= EXACT_TOL and s_rel <= EXACT_TOL
    print(f"[A] recurrent vs chunked over T={T}: o {o_rel:.3e}, final_state {s_rel:.3e} "
          f"(tol {EXACT_TOL:.0e}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_handoff() -> bool:
    """B: a prefill's last state must carry a decode forward exactly.

    Run the chunked path over T + CHUNK tokens; separately run it over the first
    T, take the `final_state` an inference cache would keep, and walk the
    remaining CHUNK tokens recurrently from it. The two must agree on those
    rows. This is the one gate that exercises the interface the decode path depends on, and a
    layout or decay-convention error shows here and nowhere else.
    """
    st = _inputs(T + CHUNK)
    whole = reference.delta_rule(st["q"], st["k"], st["v"], st["beta"], st["g"], CHUNK)

    head = {key: value[:T] for key, value in st.items()}
    prefill = reference.delta_rule(head["q"], head["k"], head["v"], head["beta"],
                                   head["g"], CHUNK)
    carried = prefill["final_state"]

    tail = {key: value[T:] for key, value in st.items()}
    o_tail, _ = reference.recurrent_delta_rule(tail["q"], tail["k"], tail["v"],
                                               tail["beta"], tail["g"],
                                               initial_state=carried)
    rel = _rel(o_tail, whole["o"][T:])
    ok = rel <= EXACT_TOL
    print(f"[B] prefill({T}) -> decode({CHUNK}) hand-off vs chunked({T + CHUNK}): "
          f"{rel:.3e} (tol {EXACT_TOL:.0e}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_against_transformers() -> bool:
    """C: the same output as `torch_recurrent_gated_delta_rule` itself.

    Their function expands q and k to H heads by `repeat_interleave` where we
    index-map, takes `[B, T, H, D]` where we take `[T, H, D]`, and **always**
    scales q by `D^-0.5` whatever `use_qk_l2norm_in_kernel` says. Our inputs
    carry no such scale, so the factor is applied to ours here -- that scale is
    the one piece of the chain `reference.py` historically did not carry.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        print(f"[C] SKIP: transformers' qwen3_5 module is unavailable ({exc})")
        return True

    st = _inputs()
    grp = _GEOMETRY[0] // _GEOMETRY[1]
    q_exp = st["q"].repeat_interleave(grp, dim=1)[None].float()
    k_exp = st["k"].repeat_interleave(grp, dim=1)[None].float()
    theirs, their_state = torch_recurrent_gated_delta_rule(
        q_exp, k_exp, st["v"][None].float(),
        g=st["g"][None].float(), beta=st["beta"][None].float(),
        output_final_state=True, use_qk_l2norm_in_kernel=False,
    )
    ours, our_state = reference.recurrent_delta_rule(st["q"], st["k"], st["v"],
                                                     st["beta"], st["g"])
    scale = D ** -0.5
    o_rel = _rel(ours * scale, theirs[0])
    s_rel = _rel(our_state, their_state[0])
    # fp32 on their side against float64 on ours, so this is a dtype floor and
    # not an exactness claim.
    tol = 1e-5
    ok = o_rel <= tol and s_rel <= tol
    print(f"[C] ours vs torch_recurrent_gated_delta_rule: o {o_rel:.3e}, "
          f"state {s_rel:.3e} (tol {tol:.0e}, their fp32) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_module_decode(steps: int = 128) -> bool:
    """D: the same output as the real module's own decode path, over a rollout.

    A and C check the recurrent *step*; this checks the model's decode *path*.
    `test_model_reference.py` runs the official class with `use_cache=False`,
    which exercises the chunked branch, so nothing so far has seen the module
    decode. Here the real `Qwen3_5ForConditionalGeneration` prefills a prompt
    with `use_cache=True` and then takes one token at a time, and
    `torch_recurrent_gated_delta_rule` is wrapped to record what the module
    hands it and what it returns. Ours is then run on exactly those inputs.

    This is also the long-rollout gate: the state accumulates over every token,
    so the worst step is reported, not the first.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError as exc:
        print(f"[D] SKIP: transformers' qwen3_5 module is unavailable ({exc})")
        return True

    import test_model_reference as tmr

    cfg = tmr.tiny_config()
    model = tmr.transformers_model(cfg, seed=7, dtype=torch.float32)
    original = modeling_qwen3_5.torch_recurrent_gated_delta_rule
    calls = []

    def recording(query, key, value, g, beta, initial_state=None, **kwargs):
        # Clone the inputs BEFORE delegating. The module hands `initial_state`
        # straight from its cache and then writes the new state back over it, so
        # a clone taken after the call holds the OUTPUT state and every
        # comparison after the first step is against the wrong seed.
        captured = dict(q=query.detach().clone(), k=key.detach().clone(),
                        v=value.detach().clone(), g=g.detach().clone(),
                        beta=beta.detach().clone(),
                        init=None if initial_state is None else initial_state.detach().clone())
        out, state = original(query, key, value, g, beta,
                              initial_state=initial_state, **kwargs)
        captured["out"] = out.detach().clone()
        calls.append(captured)
        return out, state

    modeling_qwen3_5.torch_recurrent_gated_delta_rule = recording
    try:
        prompt = torch.arange(16) % (cfg.vocab_size - 8)
        with torch.no_grad():
            step = model(input_ids=prompt[None], use_cache=True)
            for i in range(steps):
                token = torch.tensor([[int(i) % (cfg.vocab_size - 8)]])
                step = model(input_ids=token, use_cache=True,
                             past_key_values=step.past_key_values)
    finally:
        modeling_qwen3_5.torch_recurrent_gated_delta_rule = original

    if not calls:
        print("[D] FAIL: the module never took the recurrent branch")
        return False

    d = cfg.linear_key_head_dim
    scale = d ** -0.5
    worst, worst_at = 0.0, -1
    for i, c in enumerate(calls):
        # The module passes q and k pre-norm with use_qk_l2norm_in_kernel=True,
        # already repeat_interleaved to H heads, so Hg == H and grp == 1 here.
        # Normalise with `reference.l2norm`, not `F.normalize`: their `l2norm`
        # puts eps = 1e-6 INSIDE the sqrt where `F.normalize` clamps at 1e-12,
        # and at this config's q row norms of ~0.008 that is a ~1% difference,
        # not a rounding one.
        q = reference.l2norm(c["q"][0]) * scale
        k = reference.l2norm(c["k"][0])
        init = None if c["init"] is None else c["init"][0]
        ours, _ = reference.recurrent_delta_rule(q, k, c["v"][0], c["beta"][0],
                                                 c["g"][0], initial_state=init)
        rel = _rel(ours, c["out"][0])
        if rel > worst:
            worst, worst_at = rel, i
    tol = 1e-5
    ok = worst <= tol
    print(f"[D] ours vs the module's decode path over {len(calls)} recurrent calls: "
          f"worst {worst:.3e} at call {worst_at} (tol {tol:.0e}, module fp32) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_real_weight_ranges() -> bool:
    """E: the two forms must still agree on inputs a trained layer produces.

    A, C and D all draw from `make_inputs`. A synthetic draw is not the thing
    that broke `solve_tril` -- that passed its own gate at 2.325e-07 for weeks
    while the composed block returned NaN on the trained layer, which is why R1
    exists. This runs the real layer-0 weights through the block chain, takes
    the q / k / v / beta / g it produces, and compares the two forms on those.
    """
    import pathlib

    # Anchored to the repository, not the working directory: a relative path here
    # makes the gate SKIP when the suite is run from this directory rather than the
    # root, which is a silent loss of the only real-weight gate in the file.
    weights_path = (pathlib.Path(__file__).resolve().parents[2]
                    / "build_output/qwen3_8_27b/linear_attn_layer0.pt")
    if not weights_path.exists():
        print(f"[E] SKIP: {weights_path} is absent "
              "(fetch with weights.py --fetch --layer 0)")
        return True

    w = torch.load(weights_path, weights_only=True)
    t = 128
    x = reference.make_block_inputs(t, QWEN3_8_27B)
    st = reference.block(x, w, QWEN3_8_27B, CHUNK)
    o_rec, state = reference.recurrent_delta_rule(st["q"], st["k"], st["v"],
                                                  st["beta"], st["g"])
    o_rel = _rel(o_rec, st["o"])
    s_rel = _rel(state, st["final_state"])
    ok = o_rel <= EXACT_TOL and s_rel <= EXACT_TOL
    print(f"[E] real layer 0, T={t}: o {o_rel:.3e}, final_state {s_rel:.3e} "
          f"(tol {EXACT_TOL:.0e}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_kernel_golden() -> bool:
    """F: `decode_delta_rule`'s host golden, rolled out, must equal the oracle.

    The device kernel is gated against that golden on a batch of tokens; this gates
    the golden against the oracle over a rollout on one pool row, which is the only
    thing that exercises the recurrence as a recurrence. The two hold the state in
    different layouts on purpose -- the oracle's is `[H, K, V]`, the kernel's
    `[H, V, K]` -- and D is 128 on both axes, so nothing but this comparison can catch
    a transposed state. The control at the end proves the comparison is not blind to
    it.

    Runs at the model's real head geometry, which is where the GQA index mapping and
    the per-head scalar gate both matter.
    """
    import decode_delta_rule as ddr

    tokens, pool_rows, row = 24, 3, 2
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(tokens, ddr.HG, ddr.D, generator=gen).to(torch.bfloat16)
    k = torch.randn(tokens, ddr.HG, ddr.D, generator=gen).to(torch.bfloat16)
    v = torch.randn(tokens, ddr.H, ddr.D, generator=gen).to(torch.bfloat16)
    beta = torch.rand(tokens, ddr.H, generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn(tokens, ddr.H, generator=gen))
    pool = torch.randn(pool_rows, ddr.H, ddr.D, ddr.D, generator=gen).double() * 0.3
    rows = torch.tensor([row], dtype=torch.int32)

    state = pool.clone()
    out = torch.zeros(tokens, ddr.H, ddr.D, dtype=torch.float64)
    for t in range(tokens):
        step, state = ddr.golden_step(q[t:t + 1], k[t:t + 1], v[t:t + 1],
                                      beta[t:t + 1], g[t:t + 1], state, rows)
        out[t] = step[0]

    # The kernel normalises q and k itself; the oracle takes them normalised.
    qn = torch.stack([reference.l2norm(q[i], eps=ddr.L2_EPS) * ddr.Q_SCALE
                      for i in range(tokens)])
    kn = torch.stack([reference.l2norm(k[i], eps=ddr.L2_EPS) for i in range(tokens)])
    o_ref, s_ref = reference.recurrent_delta_rule(
        qn, kn, v.double(), beta.double(), g.double(),
        initial_state=ddr.golden_state_to_kv(pool[row]))

    o_rel = _rel(out, o_ref)
    s_rel = _rel(ddr.golden_state_to_kv(state[row]), s_ref)
    untouched = max(float((state[r] - pool[r]).abs().max())
                    for r in range(pool_rows) if r != row)
    # Control: the same oracle fed the UNtransposed state must disagree, or the gate
    # proves nothing about the layout.
    o_flat, _ = reference.recurrent_delta_rule(qn, kn, v.double(), beta.double(),
                                               g.double(), initial_state=pool[row])
    blind = _rel(out, o_flat) <= 1e-3

    ok = o_rel <= EXACT_TOL and s_rel <= EXACT_TOL and untouched == 0.0 and not blind
    print(f"[F] decode_delta_rule golden rolled out over T={tokens} at H={ddr.H}, "
          f"HG={ddr.HG}, D={ddr.D}: o {o_rel:.3e}, final_state {s_rel:.3e}, "
          f"other pool rows moved {untouched:.3e} (tol {EXACT_TOL:.0e}), "
          f"transpose control {'BLIND' if blind else 'detects it'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_conv_decode() -> bool:
    """G: `reference.recurrent_short_conv` three ways, like A/B/C for the delta rule.

    The conv's decode state is the only other thing a decode step carries, and it is
    a different kind of state from the delta rule's: it holds raw stream rows, so the
    prefill hand-off is a slice of the conv's own input rather than something a
    prefill kernel has to emit. All three gates below turn on that being true.
    """
    c = QWEN3_8_27B.qkv_width
    k = QWEN3_8_27B.linear_conv_kernel_dim
    t = 32
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(t, c, generator=gen, dtype=torch.float32)
    w = torch.randn(c, k, generator=gen, dtype=torch.float32) * 0.3

    # G1: from a zero state the recurrent form must equal the chunked one.
    chunked = reference.short_conv(x, w)
    stepped, state = reference.recurrent_short_conv(x, w)
    g1 = _rel(stepped, chunked)

    # G2: the hand-off. Prefill the first `head` tokens, carry the last K-1 STREAM rows,
    # and step the rest; the result must match the whole-sequence prefill on those rows.
    head = 24
    carried = x[head - (k - 1) : head]
    tail, _ = reference.recurrent_short_conv(x[head:], w, initial_state=carried)
    g2 = _rel(tail, chunked[head:])

    # G3: against transformers' own update function, which keeps the state
    # channel-major -- the transpose is the whole point of the comparison.
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_update
    except ImportError:
        print("[G] SKIP: transformers' qwen3_5 modeling module is unavailable")
        return True
    weight_param = torch.nn.Parameter(w.clone())
    # Their conv_state is [B, C, K-1]: batched and channel-major, both.
    their_state = carried.t().contiguous().unsqueeze(0).clone()
    theirs = torch.zeros(t - head, c, dtype=torch.float64)
    for i in range(t - head):
        step = x[head + i].reshape(1, c, 1)
        out = causal_conv1d_update(step, their_state, weight_param, None, "silu")
        theirs[i] = out.reshape(c).double()
    g3 = _rel(tail, theirs)
    # Their state is channel-major, so it must now equal ours transposed.
    _, ours_state = reference.recurrent_short_conv(x[head:], w, initial_state=carried)
    g3_state = _rel(their_state[0].t().double(), ours_state)

    ok = g1 <= EXACT_TOL and g2 <= EXACT_TOL and g3 <= 1e-5 and g3_state <= 1e-5
    print(f"[G] conv decode, C={c} K={k}: vs chunked {g1:.3e}, hand-off {g2:.3e} "
          f"(tol {EXACT_TOL:.0e}), vs causal_conv1d_update o {g3:.3e} state {g3_state:.3e} "
          f"(tol 1e-05, their fp32) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_attention_decode() -> bool:
    """H: paged decode attention must equal the chunked causal form's last row.

    The 16 full-attention layers carry the other decode state, and it is a third
    kind again: not a recurrence like the delta rule, not a short window like the
    convolution, but a cache that grows with the context and is addressed through a
    block table. All three gates below turn on the paging being modelled rather
    than flattened -- a contiguous cache would hide exactly the indexing error this
    is for.
    """
    hq = QWEN3_8_27B.num_attention_heads
    hkv = QWEN3_8_27B.num_key_value_heads
    hd = QWEN3_8_27B.head_dim
    page = 128
    t = 320                      # 2.5 pages: the last one is deliberately partial
    scale = hd ** -0.5
    gen = torch.Generator().manual_seed(17)
    q = torch.randn(t, hq * hd, generator=gen, dtype=torch.float32)
    k = torch.randn(t, hkv * hd, generator=gen, dtype=torch.float32)
    v = torch.randn(t, hkv * hd, generator=gen, dtype=torch.float32)

    full = reference.causal_gqa_attention(q, k, v, hq, hkv, hd, scale=scale)

    # H1: the same answer through the paged path, with the pages placed away from
    # physical zero so a kernel ignoring the block table cannot pass.
    k_pages, v_pages, table = reference.pages_from_contiguous(k, v, page, first_page=3)
    lens = torch.tensor([t], dtype=torch.int32)
    paged = reference.decode_gqa_attention(q[t - 1 : t], k_pages, v_pages, table, lens,
                                           hq, hkv, hd, page, scale=scale)
    h1 = _rel(paged, full[t - 1 : t])

    # H2: a shorter sequence in the same pages -- the tail of the last page is
    # cached but not yet real, and must not be attended to.
    short = 200
    paged_short = reference.decode_gqa_attention(q[short - 1 : short], k_pages, v_pages,
                                                 table, torch.tensor([short], dtype=torch.int32),
                                                 hq, hkv, hd, page, scale=scale)
    full_short = reference.causal_gqa_attention(q[:short], k[:short], v[:short],
                                                hq, hkv, hd, scale=scale)
    h2 = _rel(paged_short, full_short[short - 1 : short])

    # H3: control -- a block table pointing at the wrong pages must disagree, or
    # H1 proves nothing about the indexing.
    wrong = table.clone()
    wrong[0, 0], wrong[0, 1] = wrong[0, 1], wrong[0, 0]
    scrambled = reference.decode_gqa_attention(q[t - 1 : t], k_pages, v_pages, wrong, lens,
                                               hq, hkv, hd, page, scale=scale)
    h3 = _rel(scrambled, full[t - 1 : t])

    ok = h1 <= EXACT_TOL and h2 <= EXACT_TOL and h3 > 1e-3
    print(f"[H] paged decode attention, HQ={hq} HKV={hkv} D={hd} page={page}: "
          f"full T={t} {h1:.3e}, partial last page {h2:.3e} (tol {EXACT_TOL:.0e}), "
          f"scrambled block table {h3:.3e} "
          f"({'detects it' if h3 > 1e-3 else 'BLIND'}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted and ignored: this check is host-only, but CI invokes every runnable
    # file in a model directory with the common platform and device flags.
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted for CI's uniform invocation; unused")
    parser.add_argument("--steps", type=int, default=128, help="decode steps the module rollout walks")
    args = parser.parse_args()

    reference_setup = getattr(reference, "host_torch_setup", None)
    if callable(reference_setup):
        reference_setup()
    else:
        torch.set_num_threads(min(torch.get_num_threads(), 32))
        torch.backends.mkldnn.enabled = False

    global _GEOMETRY
    results = []
    for h, hg in ((H, HG), (REAL_H, REAL_HG)):
        _GEOMETRY = (h, hg)
        print(f"\nshapes: T={T} H={h} HG={hg} D={D} CHUNK={CHUNK}")
        results += [check_against_chunked(), check_handoff(), check_against_transformers()]

    # D through G do not take a head geometry: D uses the tiny config's own, E the
    # model's real one through the block chain, F the kernel's own constants, G
    # is the convolution, which has channels rather than heads, and H is the
    # full-attention path, which has its own.
    print()
    results += [check_module_decode(args.steps), check_real_weight_ranges(),
                check_kernel_golden(), check_conv_decode(),
                check_attention_decode()]
    passed = all(results)
    print("\nPASS" if passed else "\nFAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
