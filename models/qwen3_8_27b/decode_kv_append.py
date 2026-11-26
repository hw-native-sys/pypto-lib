# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Write one decode token's K and V into the paged cache `decode_attention` reads.

`decode_attention` takes `k_pages` and `v_pages` read-only and attends over a cache
somebody else filled. This is that somebody: per sequence, one row of `KV_WIDTH` BF16 into
the slot the block table names.

**The slot is derived exactly as the reader derives its pages**, which is the whole
correctness content of this kernel:

    pos  = seq_lens[i] - 1          # seq_lens counts the token being decoded
    phys = block_table[i, pos // PAGE]
    row  = phys * PAGE + pos % PAGE

`seq_lens[i]` includes the new token, because that is what `decode_attention` is given --
its page count is `(length + PAGE - 1) // PAGE` and its tail mask is indexed by the valid
count within the last page. Passing a length that excluded the new token would make the
reader skip the row this kernel just wrote, and nothing in either signature would notice.

One task per sequence. A task writes `2 * KV_WIDTH` BF16, 4 KiB in total, so at any decode
batch this kernel is its own launch cost and nothing else -- `[1, 1024]` BF16 is 2048 bytes,
well clear of the 32-byte row minimum, and each task's row is 2048-byte aligned, so two
sequences cannot share a cache line however their slots fall.
"""
import pypto.language as pl

from config import KV_PAGE_SIZE, QWEN3_8_27B

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")                 # decode sequences, one token each
KVROWS_DYN = pl.dynamic("KVROWS_DYN")       # rows of the page pool

# model config
HKV = QWEN3_8_27B.num_key_value_heads
D = QWEN3_8_27B.head_dim
KV_WIDTH = HKV * D                          # one token's K or V, flat
PAGE = KV_PAGE_SIZE                         # rows per page, the cache's addressing unit

# validation
# one sequence, then a full padded batch; a task is one sequence.
DEFAULT_CASES = (1, 16)


# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


def build_kernel(max_pages: int, page: int = PAGE, inline: bool = False):
    """The append for a cache of `max_pages` pages per sequence."""

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_decode_kv_append(
        k: pl.Tensor[[B_DYN, KV_WIDTH], pl.BF16],
        v: pl.Tensor[[B_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` counts the token being written; it must be at least 1."""
        k.bind_dynamic(0, B_DYN)
        v.bind_dynamic(0, B_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        b_dim = pl.tensor.dim(k, 0)
        for i in pl.spmd(b_dim, name_hint="decode_kv_append"):
            pos = pl.cast(pl.read(seq_lens, [i]), pl.INDEX) - 1
            p = pos // page
            phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
            row = phys * page + (pos - p * page)
            k_pages[row : row + 1, :] = k[i : i + 1, :]
            v_pages[row : row + 1, :] = v[i : i + 1, :]
        return k_pages, v_pages

    return _gdn_decode_kv_append


def build_fused_kernel(max_pages: int, page: int = PAGE, inline: bool = False):
    """The same append with v read out of the fused projection output.

    k arrives narrow because it has been through the norm and RoPE, which is what goes in
    the cache; v is never touched between the projection and here, so fused it stays a
    channel range of `[B, ATTN_QKV_WIDTH]` and no pass copies it out. The slot arithmetic is
    the one above, unchanged, and the offset comes from `decode_projection`.
    """
    from decode_projection import ATTN_QKV_WIDTH, QKV_V_BASE

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_decode_kv_append_fused(
        k: pl.Tensor[[B_DYN, KV_WIDTH], pl.BF16],
        qkv: pl.Tensor[[B_DYN, ATTN_QKV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` counts the token being written; it must be at least 1."""
        k.bind_dynamic(0, B_DYN)
        qkv.bind_dynamic(0, B_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        b_dim = pl.tensor.dim(k, 0)
        for i in pl.spmd(b_dim, name_hint="decode_kv_append_fused",
                         allow_early_resolve=EARLY_RESOLVE):
            pos = pl.cast(pl.read(seq_lens, [i]), pl.INDEX) - 1
            p = pos // page
            phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
            row = phys * page + (pos - p * page)
            k_pages[row : row + 1, :] = k[i : i + 1, :]
            v_pages[row : row + 1, :] = qkv[i : i + 1, QKV_V_BASE : QKV_V_BASE + KV_WIDTH]
        return k_pages, v_pages

    return _gdn_decode_kv_append_fused


def build_tensor_specs(batch: int = 16, max_pages: int = 8, page: int = PAGE,
                       first_page: int = 3):
    """A pool with the sequences' pages deliberately not in index order.

    `first_page` offsets every sequence's pages, so a kernel that ignored the block table
    and wrote `i * page + pos` would land on the wrong rows and the gate would catch it.
    """
    import torch
    from golden import TensorSpec

    gen = torch.Generator().manual_seed(42)
    pages_total = first_page + batch * max_pages
    rows = pages_total * page
    # Each sequence gets a scattered run of pages, not a contiguous one.
    perm = torch.randperm(batch * max_pages, generator=gen) + first_page
    table = perm.reshape(batch, max_pages).to(torch.int32)
    # Lengths that land in different pages and at different offsets, including a page
    # boundary (the first slot of a page) and a page end.
    lens = torch.tensor([1 + (i * 97) % (max_pages * page) for i in range(batch)],
                        dtype=torch.int32)
    lens[0] = 1                                 # the very first token of a sequence
    if batch > 1:
        lens[1] = page                          # the last slot of page 0
    if batch > 2:
        lens[2] = page + 1                      # the first slot of page 1
    if batch > 3:
        lens[3] = max_pages * page              # the last slot of the last page

    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return lambda: torch.randn(batch, KV_WIDTH, generator=g).to(torch.bfloat16)

    def pool(seed):
        g = torch.Generator().manual_seed(seed)
        return lambda: torch.randn(rows, KV_WIDTH, generator=g).to(torch.bfloat16)

    return [
        TensorSpec("k", [batch, KV_WIDTH], torch.bfloat16, init_value=draw(1)),
        TensorSpec("v", [batch, KV_WIDTH], torch.bfloat16, init_value=draw(2)),
        TensorSpec("block_table", [batch, max_pages], torch.int32, init_value=lambda: table),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=lambda: lens),
        TensorSpec("k_pages", [rows, KV_WIDTH], torch.bfloat16, init_value=pool(3)),
        TensorSpec("v_pages", [rows, KV_WIDTH], torch.bfloat16, init_value=pool(4)),
    ]


def golden_gdn_decode_kv_append(page: int = PAGE):
    """The oracle at one page size, which must be the one the kernel was built with."""

    def golden(tensors):
        import reference

        reference.append_kv_pages(
            tensors["k_pages"], tensors["v_pages"], tensors["block_table"],
            tensors["seq_lens"], tensors["k"], tensors["v"], page,
        )

    return golden


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)

    for batch in cases:
        print(f"[case] decode_kv_append batch={batch}", flush=True)
        result = run(
            fn=build_kernel(max_pages=args.max_pages),
            specs=build_tensor_specs(batch=batch, max_pages=args.max_pages),
            golden_fn=golden_gdn_decode_kv_append(),
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
