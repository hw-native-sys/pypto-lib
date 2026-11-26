# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B paged attention, decode: one query row per sequence over a paged KV cache.

    o = softmax(q k^T) v        per query head, over every cached key

`q` arrives already q-normed, RoPE'd and scaled by `head_dim ** -0.5`, which is what
`qk_rmsnorm_rope` produces. Query head `h` reads KV head `h // (HQ // HKV)`.

**There is no causal mask.** The query is the newest position, so every cached key is in
its past.

One task owns one `(sequence, KV head)` pair and the `GROUP` query heads that share it.
`GROUP` is 6 and the cube's M fractal is 16, so the query tile is **16 rows with 6 real
ones**, declared through the slice's `valid_shape`.

The sequence length rarely lands on a page boundary, so the last page holds rows that are
cached but not yet real. `tail_mask` is indexed by the page's valid count: row `v` opens
its first `v` columns and closes the rest, and row `PAGE` -- which every full page selects
-- is all open.

**The mask is not the whole story: the rows a sequence does not own must be zero.** A page
is loaded whole into `k_tile` and `v_tile`, so an unowned row still reaches the QK matmul,
and `tail_mask` is additive -- it closes a column after its score exists. A large value
there survives into `row_max` and scales the whole row's softmax; in the PV matmul it
meets a zeroed probability and `0 * x` is not zero when `x` is not finite. Measured: with
the unowned rows at 3e38 the output of three of four sequences goes non-finite, and with
them at NaN so does more of it. This is `deepseek_v4_flash_mtp`'s contract too, whose
sparse attention reads "invalid lanes (NEG_INF bias, zero kv rows)" and whose fixtures
build those zero rows deliberately. `page_layout` does the same here, so the gate holds
the kernel to it.

`l` sums the **BF16-rounded** probabilities, not the FP32 ones, because the BF16 tile is
what the PV matmul consumes: the denominator has to be the sum of the numerator's own
terms.

`SPLITS` partitions each sequence's pages across several tasks, trading the length of the
serial online-softmax chain against a merge pass. It is **1**: at every batch
`decode_projection.check_batch` allows, the cores are already full without splitting.
"""
import pypto.language as pl

from config import KV_PAGE_SIZE, QWEN3_8_27B

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")                     # decode sequences, one query row each
KVROWS_DYN = pl.dynamic("KVROWS_DYN")           # rows of the paged KV cache

# model config
HQ = QWEN3_8_27B.num_attention_heads            # 24 query heads
HKV = QWEN3_8_27B.num_key_value_heads           # 4 KV heads
D = QWEN3_8_27B.head_dim                        # 256
GROUP = HQ // HKV                               # query heads sharing one KV head
Q_WIDTH = HQ * D
KV_WIDTH = HKV * D
PAGE = KV_PAGE_SIZE                             # rows per page, the cache's addressing unit

# validation
# one sequence, then the serving batch; a task is one sequence and KV head.
# (batch, context). The wide case carries the boundary set below across its lanes; the
# batch-1 case is the task-starved regime, where `batch * HKV` is 4 tasks on 48 cores.
DEFAULT_CASES = ((4, 8192), (1, 320))

# tiling
SPLITS = 1              # page ranges per sequence; see the table above
Q_PAD = 16              # the cube's M fractal: GROUP real rows live in a 16-row tile

MASK_NEG = -1.0e30      # additive mask; its exp underflows to exactly zero
# A one-column FP32 store has a four-byte row, under ptoas's 32-byte minimum, so
# the split partials carry `m` and `l` broadcast across this many columns.
ML_COLS = 16


def build_tail_mask(page: int = PAGE):
    """`[page + 1, page]` FP32: row `v` opens columns `< v`. Row `page` is all open."""
    import torch

    cols = torch.arange(page).reshape(1, page)
    valid = torch.arange(page + 1).reshape(page + 1, 1)
    return torch.where(cols < valid, 0.0, MASK_NEG).float()


# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


def build_kernel(max_pages: int, page: int = PAGE, splits: int = SPLITS,
                 inline: bool = False):
    """Decode attention for a cache of at most `max_pages` pages per sequence.

    `splits` partitions each sequence's pages across that many tasks, which is the only
    way to fill the machine at small batch: without it the task count is
    `batch * HKV`, so **batch 1 has four tasks** and reaches 13.5% of the copy roof
    against the 65% that batch 16 reaches with 64. Each split produces its own running
    `(m, l, acc)` and a second pass merges them, which is exact: the online-softmax
    state composes, so the merge is the same rescale the loop already does.
    """
    if page % 16:
        raise ValueError(f"page={page} must be a multiple of 16, the cube's fractal")
    if splits < 1:
        raise ValueError(f"splits={splits} must be at least 1")
    if splits > 1:
        return _build_split_kernel(max_pages, page, splits, inline)

    def _gdn_decode_attention(
        q: pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16],
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        tail_mask: pl.Tensor[[page + 1, page], pl.FP32],
        o: pl.Out[pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` must be at least 1 and at most `max_pages * page`."""
        q.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        o.bind_dynamic(0, B_DYN)
        b_dim = pl.tensor.dim(q, 0)
        # A head is one row of this view, so the GROUP heads sharing a KV head are a
        # contiguous row block and the query tile is a plain 2D slice.
        q_rows = pl.reshape(q, [b_dim * HQ, D])
        o_rows = pl.reshape(o, [b_dim * HQ, D])

        for task in pl.spmd(b_dim * HKV, name_hint="decode_attention",
                            allow_early_resolve=EARLY_RESOLVE):
            i = task // HKV
            j = task % HKV
            length = pl.cast(pl.read(seq_lens, [i]), pl.INDEX)
            pages = (length + page - 1) // page
            qr0 = i * HQ + j * GROUP
            kd0 = j * D
            q_tile = pl.slice(q_rows, [Q_PAD, D], [qr0, 0], valid_shape=[GROUP, D])

            # The running max and sum are [Q_PAD, 1] columns seeded from a reduction,
            # so they take the layout the reductions produce.
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)

            for p, (m_run, l_run, acc_run) in pl.range(
                    pages, init_values=(m_init, zero_col, acc_init)):
                phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
                r0 = phys * page
                k_tile = k_pages[r0 : r0 + page, kd0 : kd0 + D]
                scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
                # Row `page` of the table is all open, so a full page needs no branch.
                valid = pl.min(page, length - p * page)
                masked = pl.col_expand_add(scores, tail_mask[valid : valid + 1, 0:page])
                m_cur = pl.row_max(masked)
                m_new = pl.maximum(m_run, m_cur)
                alpha = pl.exp(pl.sub(m_run, m_new))
                probs = pl.exp(pl.row_expand_sub(masked, m_new))
                probs_bf = pl.cast(probs, target_type=pl.BF16, mode="rint")
                l_cur = pl.row_sum(pl.cast(probs_bf, target_type=pl.FP32))
                l_new = pl.add(pl.mul(l_run, alpha), l_cur)
                v_tile = v_pages[r0 : r0 + page, kd0 : kd0 + D]
                pv = pl.matmul(probs_bf, v_tile, out_dtype=pl.FP32)
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha), pv)
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            ctx = pl.row_expand_mul(acc_end, pl.recip(l_end, high_precision=True))
            o_rows[qr0 : qr0 + GROUP, 0:D] = pl.cast(
                pl.slice(ctx, [GROUP, D], [0, 0]), target_type=pl.BF16, mode="rint")
        return o

    return (pl.jit.inline(_gdn_decode_attention) if inline
            else pl.jit(_gdn_decode_attention))


def _build_split_kernel(max_pages: int, page: int, splits: int, inline: bool):
    """Two passes: partial softmax states per page range, then an exact merge.

    The partial `m` and `l` are `[Q_PAD, 1]` columns, and a one-column FP32 store has a
    four-byte row, which ptoas rejects on its 32-byte minimum. They are therefore stored
    broadcast across 16 columns and read back with `pl.row_max`: every column holds the
    same value, so the maximum is that value exactly, and nothing needs a width-one tile
    view.

    A split whose page range is empty -- which happens whenever `splits` exceeds a
    sequence's page count -- stores its seeds, `m = MASK_NEG` and `l = 0`. The merge then
    weights it by `exp(MASK_NEG - m*)`, which underflows to zero, so an empty split
    contributes nothing and needs no guard.
    """
    def _gdn_decode_attention_split(
        q: pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16],
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        tail_mask: pl.Tensor[[page + 1, page], pl.FP32],
        o: pl.Out[pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` must be at least 1 and at most `max_pages * page`."""
        q.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        o.bind_dynamic(0, B_DYN)
        b_dim = pl.tensor.dim(q, 0)
        q_rows = pl.reshape(q, [b_dim * HQ, D])
        o_rows = pl.reshape(o, [b_dim * HQ, D])

        part_rows = b_dim * HKV * splits * Q_PAD
        acc_part = pl.create_tensor([part_rows, D], dtype=pl.FP32)
        m_part = pl.create_tensor([part_rows, ML_COLS], dtype=pl.FP32)
        l_part = pl.create_tensor([part_rows, ML_COLS], dtype=pl.FP32)

        with pl.spmd(b_dim * HKV * splits, name_hint="decode_attention_split",
                     allow_early_resolve=EARLY_RESOLVE) as part_tid:
            task = pl.tile.get_block_idx()
            i = task // (HKV * splits)
            j = (task // splits) % HKV
            s = task % splits
            length = pl.cast(pl.read(seq_lens, [i]), pl.INDEX)
            pages = (length + page - 1) // page
            per_split = (pages + splits - 1) // splits
            p_lo = s * per_split
            n_pages = pl.max(pl.min(per_split, pages - p_lo), 0)

            qr0 = i * HQ + j * GROUP
            kd0 = j * D
            q_tile = pl.slice(q_rows, [Q_PAD, D], [qr0, 0], valid_shape=[GROUP, D])
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)

            for pp, (m_run, l_run, acc_run) in pl.range(
                    n_pages, init_values=(m_init, zero_col, acc_init)):
                p = p_lo + pp
                phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
                r0 = phys * page
                k_tile = k_pages[r0 : r0 + page, kd0 : kd0 + D]
                scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
                valid = pl.min(page, length - p * page)
                masked = pl.col_expand_add(scores, tail_mask[valid : valid + 1, 0:page])
                m_cur = pl.row_max(masked)
                m_new = pl.maximum(m_run, m_cur)
                alpha = pl.exp(pl.sub(m_run, m_new))
                probs = pl.exp(pl.row_expand_sub(masked, m_new))
                probs_bf = pl.cast(probs, target_type=pl.BF16, mode="rint")
                l_cur = pl.row_sum(pl.cast(probs_bf, target_type=pl.FP32))
                l_new = pl.add(pl.mul(l_run, alpha), l_cur)
                v_tile = v_pages[r0 : r0 + page, kd0 : kd0 + D]
                pv = pl.matmul(probs_bf, v_tile, out_dtype=pl.FP32)
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha), pv)
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            pr0 = ((i * HKV + j) * splits + s) * Q_PAD
            ones_ml = pl.full([Q_PAD, ML_COLS], dtype=pl.FP32, value=1.0)
            acc_part[pr0 : pr0 + Q_PAD, 0:D] = acc_end
            m_part[pr0 : pr0 + Q_PAD, 0:ML_COLS] = pl.row_expand_mul(ones_ml, m_end)
            l_part[pr0 : pr0 + Q_PAD, 0:ML_COLS] = pl.row_expand_mul(ones_ml, l_end)

        with pl.spmd(b_dim * HKV, name_hint="decode_attention_merge", deps=[part_tid],
                     allow_early_resolve=EARLY_RESOLVE):
            task = pl.tile.get_block_idx()
            i = task // HKV
            j = task % HKV
            base = (i * HKV + j) * splits * Q_PAD
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)
            for sp, (m_run, l_run, acc_run) in pl.range(
                    splits, init_values=(m_init, zero_col, acc_init)):
                sr0 = base + sp * Q_PAD
                # Every column holds the same value, so the maximum is that value.
                m_s = pl.row_max(pl.slice(m_part, [Q_PAD, ML_COLS], [sr0, 0]))
                l_s = pl.row_max(pl.slice(l_part, [Q_PAD, ML_COLS], [sr0, 0]))
                acc_s = pl.slice(acc_part, [Q_PAD, D], [sr0, 0])
                m_new = pl.maximum(m_run, m_s)
                alpha = pl.exp(pl.sub(m_run, m_new))
                beta = pl.exp(pl.sub(m_s, m_new))
                l_new = pl.add(pl.mul(l_run, alpha), pl.mul(l_s, beta))
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha),
                                 pl.row_expand_mul(acc_s, beta))
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            qr0 = i * HQ + j * GROUP
            ctx = pl.row_expand_mul(acc_end, pl.recip(l_end, high_precision=True))
            o_rows[qr0 : qr0 + GROUP, 0:D] = pl.cast(
                pl.slice(ctx, [GROUP, D], [0, 0]), target_type=pl.BF16, mode="rint")
        return o

    return (pl.jit.inline(_gdn_decode_attention_split) if inline
            else pl.jit(_gdn_decode_attention_split))


def golden_fn(tensors):
    """The float64 oracle, through `reference.decode_gqa_attention`.

    `q` already carries the `head_dim ** -0.5` scale, so the reference takes scale 1.
    """
    import reference

    out = reference.decode_gqa_attention(
        tensors["q"], tensors["k_pages"], tensors["v_pages"], tensors["block_table"],
        tensors["seq_lens"], HQ, HKV, D, PAGE, scale=1.0)
    tensors["o"].copy_(out.to(tensors["o"].dtype))


def owned_mask(table, lens, pool_rows: int):
    """Which pool rows the block table's lanes hold, as a `[pool_rows]` bool.

    One implementation of the zero-row contract above, so a fixture cannot drift from it.
    Lane `i` takes its length from `lens[i % len(lens)]`, which lets a stacked table --
    `decode_fwd`'s `[groups * m, max_pages]`, one length set repeated per layer group --
    use the same rule as a plain `[batch, max_pages]` one.
    """
    import torch

    owned = torch.zeros(pool_rows, dtype=torch.bool)
    n = len(lens)
    for row in range(table.shape[0]):
        length = int(lens[row % n])
        for p in range((length + PAGE - 1) // PAGE):
            base = int(table[row, p]) * PAGE
            held = min(PAGE, length - p * PAGE)
            owned[base : base + held] = True
    return owned


def page_layout(batch: int, context: int, max_pages: int, pool_pages: int | None = None):
    """Per-lane lengths, the block table for them, and the pool they need.

    Shared with `decode_gated_attention`, which attends over the same cache: one boundary
    per lane tiled across the batch, the way `deepseek_v4_flash_mtp.utils` tiles its start
    positions -- the deepest page walk, a sequence's first token, both sides of a page
    edge, and a length that leaves the last page partial. Lengths differ per sequence, as
    they do in serving.

    **The pool holds what the lanes actually need, not `batch * max_pages`.** A real
    allocator gives a short sequence one page and a long one sixty-four, and sizing for the
    deepest lane times the batch is what makes a long-context case expensive. Pages come
    off a shuffled free list, so a kernel that ignored the table and read
    `sequence * max_pages` would fail.
    """
    import torch

    pages = (context + PAGE - 1) // PAGE
    if pages > max_pages:
        raise ValueError(f"context={context} needs {pages} pages, over max_pages={max_pages}")
    pattern = [context, 1, PAGE, PAGE + 1, max(1, context - 1), min(2 * PAGE, context)]
    lens = torch.tensor([pattern[i % len(pattern)] for i in range(batch)], dtype=torch.int32)
    lane_pages = [(int(n) + PAGE - 1) // PAGE for n in lens]
    pool = sum(lane_pages) if pool_pages is None else pool_pages

    free = torch.randperm(pool, generator=torch.Generator().manual_seed(29))
    table = torch.zeros(batch, max_pages, dtype=torch.int32)
    taken = 0
    for s, pages_s in enumerate(lane_pages):
        table[s, :pages_s] = free[taken : taken + pages_s].to(torch.int32)
        taken += pages_s
    return lens, table, pool, owned_mask(table, lens, pool * PAGE)


def build_tensor_specs(batch: int = 4, context: int = 320, max_pages: int = 8,
                       pool_pages: int | None = None):
    """A batch of sequences whose pages are scattered through a pool sized to their lengths.

    `page_layout` owns the lane lengths, the table, the pool extent and which rows a
    sequence holds. The rows none holds are zeroed, which is the contract below.
    """
    import torch
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    lens, table, pool_pages, owned = page_layout(batch, context, max_pages, pool_pages)
    gen = torch.Generator().manual_seed(23)

    def pool(seed):
        def make():
            t = torch.randn(pool_pages * PAGE, KV_WIDTH, generator=torch.Generator().manual_seed(seed))
            t[~owned] = 0.0
            return t.to(bf)
        return make

    return [
        TensorSpec("q", [batch, Q_WIDTH], bf,
                   init_value=lambda: (torch.randn(batch, Q_WIDTH, generator=gen)
                                       * (D ** -0.5)).to(bf)),
        TensorSpec("k_pages", [pool_pages * PAGE, KV_WIDTH], bf, init_value=pool(31)),
        TensorSpec("v_pages", [pool_pages * PAGE, KV_WIDTH], bf, init_value=pool(32)),
        TensorSpec("block_table", [batch, max_pages], i32, init_value=lambda: table),
        TensorSpec("seq_lens", [batch], i32, init_value=lambda: lens),
        TensorSpec("tail_mask", [PAGE + 1, PAGE], f32, init_value=lambda: build_tail_mask()),
        TensorSpec("o", [batch, Q_WIDTH], bf),
    ]


if __name__ == "__main__":
    import argparse

    import torch
    from golden import run

    from utils import rel_frobenius

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--context", type=int, default=320)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="pages per sequence; derived from --context when omitted, "
                             "because the two disagreeing is only ever a mistake")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--rounds", type=int, default=0)
    args = parser.parse_args()
    cases = [(args.batch, args.context)] if args.batch else list(DEFAULT_CASES)
    torch.set_num_threads(min(torch.get_num_threads(), 32))
    torch.backends.mkldnn.enabled = False
    if args.rounds:
        import os
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for batch, context in cases:
        max_pages = args.max_pages or (context + PAGE - 1) // PAGE
        if max_pages * PAGE < context:
            parser.error(f"--max-pages {max_pages} holds {max_pages * PAGE} tokens, "
                         f"under context {context}")
        print(f"[case] decode_attention batch={batch} context={context}", flush=True)
        result = run(
            fn=build_kernel(max_pages, splits=args.splits),
            specs=build_tensor_specs(batch, context, max_pages),
            golden_fn=None if args.rounds else golden_fn,
            compile_only=args.compile_only,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=2e-2, atol=2e-2,
            compare_fn=None if args.rounds else {"o": rel_frobenius(6e-3, "o")},
        )
        if args.rounds and result.bench is not None:
            import statistics
            s = [x for x in result.bench.per_round("effective") if x > 0]
            if s:
                kv = 2 * args.context * KV_WIDTH * 2 * batch
                med = statistics.median(s)
                print(f"[bench] batch={batch} context={args.context} "
                      f"median={med:.1f} us  KV {kv / 1e6:.2f} MB -> {kv / med / 1e3:.1f} GB/s",
                      flush=True)
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
