# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""On-chip buffer budget check for one PyPTO cube (matmul) tile.

Standalone utility — no PyPTO dependency. Complements hint_l1_tile.py: that tool
ranks tile *candidates* by DMA; this one tells you, for a *chosen* tile, which
on-chip buffer it blows (or how much room is left to grow) and whether the
weight tile clears the DMA cache-line floor.

It models a `pl.matmul`/`pl.matmul_acc` cube scope. Inputs are the *tile* dims
actually fed to one matmul call (the L1-resident fragment), not the full GEMM:

  M  = row tile           (the M tile fed to one matmul)
  N  = weight N fragment  (the per-call N tile)
  K  = K fragment         (the per-call K tile)

Weight layout (the transpose lever). The B/weight can be stored two ways, and
which dim is *contiguous* in GM decides which dim the cache-line floor binds:

  --no-b-trans (default): weight stored [K, N], N-contiguous.  A `pl.matmul(a, w)`.
                          Each weight row is a run of N elems -> floor on N.
  --b-trans             : weight stored [N, K], K-contiguous.  A `pl.matmul(a, w, b_trans=True)`.
                          Each weight row is a run of K elems -> floor on K.

Transposing does NOT change the Mat/L1 footprint (the tile is N*K*bytes either
way) or the total weight bytes — it only changes which axis must be >= a cache
line, and (not modeled here) the L1->L0B load path. For a transaction-bound
MTE2 cube, storing the weight so the *long* contraction axis is contiguous
(K-contiguous via --b-trans) yields longer bursts / fewer transactions; several
DSV4 compressor kernels do exactly this. Confirm the win on device — the L0B
transpose-on-load cost is not in this model.

The three walls, in the order they usually bind:

  Acc (L0C)  >= M * N * bytes_out * n_accum                      accumulator(s), never in L1
  Mat (L1)   ~= (N * n_weights + M) * K * bytes_in * double_buf  weights + activation, double-buffered
  cache-line :  contig_dim * bytes_in >= cache_line (512 B)      else MTE2 weight DMA wastes the line
                (contig_dim = K if --b-trans else N)

Mat is the usual wall when growing N. The fix is to trade the *free* (strided)
axis down to buy room, keeping the contiguous axis >= the cache-line floor:
with --no-b-trans, N is contiguous (keep N >= floor) so trade K down; with
--b-trans, K is contiguous (keep K >= floor) so trade N down. The Mat number is
an *estimate* (real allocation adds alignment padding, ~10-25%); the device
compile's `memory_after_AllocateMemoryAddr.txt` / "Mat buffer usage ... exceeds"
is ground truth. This tool flags MARGINAL when the estimate is within 15% of the
budget.
"""
import argparse

# Per-core on-chip budgets (bytes). 910B (a2a3) defaults; a5/950 overrides Acc.
PLATFORM = {
    "a2a3": {"mat": 512 * 1024, "acc": 128 * 1024, "l0a": 64 * 1024, "l0b": 64 * 1024, "ub": 192 * 1024},
    "a5":   {"mat": 512 * 1024, "acc": 256 * 1024, "l0a": 64 * 1024, "l0b": 64 * 1024, "ub": 256 * 1024},
}


def kb(x):
    return f"{x / 1024:7.1f} KB"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--M", type=int, required=True, help="row tile (M fed to one matmul)")
    p.add_argument("--N", type=int, required=True, help="weight N fragment (per-call N tile)")
    p.add_argument("--K", type=int, required=True, help="K fragment (per-call K tile)")
    p.add_argument("--bytes-in", type=int, default=1, help="A/B elem bytes (default 1 = INT8)")
    p.add_argument("--bytes-out", type=int, default=4, help="accumulator elem bytes (default 4 = INT32/FP32)")
    p.add_argument("--weights", type=int, default=1, help="weight operands held at once (fused two-weight scope=2, single=1)")
    p.add_argument("--accum", type=int, default=None, help="live accumulators (default = --weights)")
    p.add_argument("--double-buffer", type=int, default=2, help="pipeline depth (default 2)")
    p.add_argument("--cache-line", type=int, default=512, help="L2 cache line bytes (default 512)")
    p.add_argument("--b-trans", dest="b_trans", action="store_true",
                   help="weight stored [N, K] K-contiguous (pl.matmul b_trans=True); "
                        "cache-line floor binds on K. Default: weight [K, N] N-contiguous, floor on N.")
    p.add_argument("--platform", choices=PLATFORM, default="a2a3")
    p.add_argument("--mat-bytes", type=int, default=None, help="override L1/Mat budget")
    p.add_argument("--acc-bytes", type=int, default=None, help="override L0C/Acc budget")
    a = p.parse_args()

    # Guard against zero/negative args -- K and the budgets are divisors below.
    for name, val in (("M", a.M), ("N", a.N), ("K", a.K), ("bytes-in", a.bytes_in),
                      ("bytes-out", a.bytes_out), ("weights", a.weights),
                      ("double-buffer", a.double_buffer)):
        if val <= 0:
            p.error(f"--{name} must be > 0 (got {val})")
    for name, val in (("accum", a.accum), ("mat-bytes", a.mat_bytes), ("acc-bytes", a.acc_bytes)):
        if val is not None and val <= 0:
            p.error(f"--{name} must be > 0 (got {val})")

    n_accum = a.accum if a.accum is not None else a.weights
    bud = dict(PLATFORM[a.platform])
    if a.mat_bytes:
        bud["mat"] = a.mat_bytes
    if a.acc_bytes:
        bud["acc"] = a.acc_bytes

    acc = a.M * a.N * a.bytes_out * n_accum
    weight_l1 = a.N * a.K * a.bytes_in * a.weights * a.double_buffer
    act_l1 = a.M * a.K * a.bytes_in * a.double_buffer
    mat = weight_l1 + act_l1
    # Contiguous weight axis depends on the storage layout (the transpose lever).
    contig_name = "K" if a.b_trans else "N"
    contig_dim = a.K if a.b_trans else a.N
    contig = contig_dim * a.bytes_in

    def verdict(used, budget):
        if used > budget:
            return "OVER"
        if used > 0.85 * budget:
            return "MARGINAL"
        return "fit"

    layout = "[N,K] K-contig (b_trans)" if a.b_trans else "[K,N] N-contig"
    print(f"# cube tile  M={a.M} N={a.N} K={a.K}  bytes_in={a.bytes_in} bytes_out={a.bytes_out}"
          f"  weights={a.weights} accum={n_accum} dbuf={a.double_buffer}  weight={layout}  [{a.platform}]")
    print(f"  Acc (L0C)   {kb(acc)} / {kb(bud['acc'])}   {verdict(acc, bud['acc'])}"
          f"   = M*N*{a.bytes_out}*{n_accum}")
    print(f"  Mat (L1)    {kb(mat)} / {kb(bud['mat'])}   {verdict(mat, bud['mat'])}"
          f"   = weight {kb(weight_l1)} + act {kb(act_l1)}  (estimate; +10-25% alignment on device)")
    cl = "OK" if contig >= a.cache_line else f"UNDER ({contig}B < {a.cache_line}B -> MTE2 wastes line)"
    print(f"  cache-line  {contig_name}*bytes_in = {contig} B   {cl}   (contig axis = {contig_name})")

    # Headroom / suggestions. The cache-line floor binds on the *contiguous* axis
    # (K under --b-trans, else N); the *strided* axis is the one you trade to
    # escape a Mat wall.
    cl_elems = a.cache_line // a.bytes_in  # min contiguous elems to fill a line
    print("  ---")
    if acc > bud["acc"]:
        print("  Acc is the wall: drop N or M, or split accumulators.")
    if mat > bud["mat"]:
        if act_l1 >= bud["mat"]:
            print("  Mat is the wall: activation alone exceeds the L1 budget -- reduce M or K.")
        else:
            # Largest N that fits Mat at this K, and largest K at this N.
            n_max = (bud["mat"] / a.double_buffer / a.bytes_in - a.M * a.K) / (a.weights * a.K)
            k_max = bud["mat"] / a.double_buffer / a.bytes_in / (a.N * a.weights + a.M)
            if a.b_trans:
                # K is contiguous -> keep K on a cache-line multiple; N is free (16-multiple).
                n_fit = max(0, int(n_max // 16 * 16))
                k_fit = max(0, int(k_max // a.cache_line * a.cache_line))
            else:
                # N is contiguous -> keep N >= cache-line floor; K is free (16-multiple).
                n_fit = max(0, int(n_max // 16 * 16))
                k_fit = max(0, int(k_max // 16 * 16))
            print(f"  Mat is the wall: at K={a.K}, N<={n_fit} fits; at N={a.N}, K<={k_fit} fits.")
            if a.b_trans:
                print(f"  -> weight is K-contiguous: keep K>={a.cache_line}B (K>={cl_elems} elems), "
                      f"trade N down to {n_fit} to keep K={a.K}, or shrink K to {k_fit}.")
            else:
                print(f"  -> weight is N-contiguous: keep N>={a.cache_line}B (N>={cl_elems} elems), "
                      f"trade K down to {k_fit} to keep N={a.N}, or shrink N to {n_fit} (>= {cl_elems}).")
    if acc <= bud["acc"] and mat <= bud["mat"] and contig >= a.cache_line:
        n_room = int((bud["mat"] / a.double_buffer / a.bytes_in - a.M * a.K) / (a.weights * a.K) // 16 * 16)
        acc_n = int(bud["acc"] / (a.M * a.bytes_out * n_accum) // 16 * 16)
        print(f"  fits. N can grow to ~{min(n_room, acc_n)} at this K "
              f"(Mat caps N<= {n_room}, Acc caps N<= {acc_n}) before a wall — verify on device.")


if __name__ == "__main__":
    main()
