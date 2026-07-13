# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hint matmul tile sizes with 3-level decomposition and pin exploration.

Standalone utility — no PyPTO dependency.

3-level decomposition per axis:
  M = OM (outer, cross-core)  ×  LM (loop, sequential on one core)  ×  TM (tile, L1 fragment)
  N = ON                       ×  LN                                  ×  TN
  K = OK                       ×  LK                                  ×  TK

Tile sizes (TM, TN, TK) are constrained to multiples of `tile_multiple`
(default 16, the matmul fragment alignment). Each dim is padded up to the
next multiple; the padding shows up as waste %.

Each outer (OM × ON × OK) iteration is dispatched as one independent task
across cores. Within a core, the LM × LN × LK loop iterates over tiles.
A tile is the L1-resident fragment for one load/store.

Only A and B occupy L1. C is the matmul accumulator: it lives in a separate
on-chip accumulator buffer (Ascend L0C), never in L1, and is therefore always
"pinned" there — see the Acc-fit section below.

Pin modes for A / B (independently auto-explored per (tile, outer) combo):
  - Pin A: hold the entire outer-task A slice (LM*TM × LK*TK × ba) in L1.
           A is invariant across `ln`, so pinning saves an LN-factor of
           A reloads per task. L1 cost grows from one tile to the slice.
  - Pin B: hold the outer-task B slice (LK*TK × LN*TN × bb) in L1; saves
           an LM-factor of B reloads.

C is always accumulator-resident: each TM×TN output fragment is reduced over
K in the Acc buffer (no C reload per K iteration) and written to DDR once,
after its K loop completes. For split-K (OK > 1) the partial fragment is
atomic-added to DDR at the end of each outer task. C costs L1 nothing; its
only constraint is that one fragment fits in the Acc buffer.

Pin is a real choice: pinning a large operand often exceeds L1. The search
sweeps all 2^2 A/B pin subsets and keeps the feasible ones.

L1 fit (A + B only; must be ≤ l1_available_bytes; post-double-buffer accounting):
  Pinned operands are held resident across the inner loop → single buffer.
  Non-pinned operands cycle through L1 each loop iteration → double-buffered
  (2× tile) to overlap load/store with compute. Double-buffering is only
  applied when there is actually an inner loop (LM*LN*LK > 1).

  per-operand contribution = pin_size if pinned
                           = (2 * tile_size if LM*LN*LK > 1 else tile_size)

  A_tile = TM*TK*ba         A_pin = LM*TM*LK*TK*ba
  B_tile = TK*TN*bb         B_pin = LK*TK*LN*TN*bb

Acc fit (C_acc = TM*TN*bc vs acc_buffer_bytes; the L0C accumulator, default 128 KB
on 910B / 256 KB on a5):
  One TM×TN output fragment reduces in the accumulator across its K loop; bc is the
  accumulator element size (FP32 = 4 on the cube, regardless of the DDR output dtype).

  C_acc = TM*TN*bc

  Unlike L1, L0C is a *soft* wall. When C_acc exceeds it, the pypto compiler's
  AutoTileMatmulL0 pass auto-tiles the output into L0C-fitting sub-tiles (it does not
  fail), at a per-output-tile FIXPIPE L0C->L1 drain — 910B ~245 cycles fixed +
  bc*m*n/118 B/cyc, fully exposed (no L0C double-buffer emitter yet). This tool keeps a
  hard C_acc <= acc_buffer_bytes skip as the conservative "fits one L0C tile / no
  compiler output-tiling" view; candidates above it are feasible but pay ~(G-1)*245
  cycles (G = ceil(C_acc / acc_buffer_bytes)) that this DMA ranker does not model. Auto-
  tiling is deferred — Acc then hard — for matmul_acc, a Vec/PV left operand, or a mixed
  on-chip consumer (see the AutoTileMatmulL0 pass doc in the pypto repo).

DMA cost (no inter-task reuse; per-tile load/store, summed across iterations):
  Each load's innermost-row stride is rounded up to `cache_line_bytes`
  (default 512 B on Ascend 910B L2). A tile with TN*bb = 256 B per row still
  pulls 512 B per row from DDR, doubling its B-load cost. Sub-cache-line
  tiles are allowed but charged the full cache line.

  row_a_eff = max(TK*ba, cache_line)            # bytes per A row  (A=[M,K])
  row_c_eff = max(TN*bc, cache_line)            # bytes per C row  (C=[M,N])

  A_tile_eff = TM * row_a_eff
  C_tile_eff = TM * row_c_eff

  The B tile depends on the weight layout (``b_trans``, default False):
    b_trans=False  B=[K,N], N-contiguous:  B_tile_eff = TK * max(TN*bb, line)
    b_trans=True   B=[N,K], K-contiguous:  B_tile_eff = TN * max(TK*bb, line)
  LLM weight matmuls use the transposed (K-contiguous) layout, where a short K
  tile leaves TK*bb below the cache line and doubles the B load — pass
  ``b_trans=True`` (CLI ``--b-trans``) to model that; the default keeps the
  plain row-major B for backward compatibility.

  DMA(A) = num_tasks * (LM*LK if pin A else LM*LN*LK) * A_tile_eff
  DMA(B) = num_tasks * (LN*LK if pin B else LM*LN*LK) * B_tile_eff
  DMA(C) = num_tasks * (LM*LN) * C_tile_eff   # accumulator: one write per fragment
  Total  = DMA(A) + DMA(B) + DMA(C)

Two waste metrics are reported per candidate (both informational — neither
filters; ranking is by DMA cost, which already prices padding in):
  - MAC waste = m_eff*n_eff*k_eff / (M*N*K) - 1
      Wasted *compute*: padding each dim up to the tile multiple (plus any
      O*L*T overshoot). It is a single volume ratio, so a sub-fragment M
      (e.g. M=8 padded to 16) reads as ~100% — the whole M*N*K volume doubles.
  - DMA waste = actual_dma / useful_dma - 1   (the important one for bandwidth)
      Wasted *bytes moved* vs the same tiling strategy costed with real
      (unpadded) dims and no cache-line rounding. Unlike MAC waste it is
      per-operand: A is [M,K], B is [K,N], C is [M,N], so padding M inflates
      only A and C — B has no M axis and its load is untouched. It also charges
      the 512 B cache-line rounding (a sub-line TK*ba or TN*bb row pays the full
      line). For a weight-dominated decode GEMM (tiny M, huge B) DMA waste stays
      small even when MAC waste is 100%.

Wall-clock proxy: tasks are dispatched in waves of `cores` (default 24).
A 25-task config takes 2 waves on 24 cores — almost 2× slower than a
24-task config with similar per-task work. The cost we minimize is therefore

  sequential_dma = per_task_dma * ceil(num_tasks / cores)
                 = (total_dma / num_tasks) * ceil(num_tasks / cores)

This is the DMA along the critical path (slowest-wave-of-each-batch),
not the total bytes moved. Total DMA is reported too as a secondary metric.

Wave cap: configs with more than `max_waves` waves are discarded (default 3).
Many-wave configs accumulate per-task dispatch overhead the model doesn't
explicitly charge, so capping waves favors in-core loop solutions over
many small tasks. Loosen with `--max-waves` for problems that genuinely
need more parallelism.

Objective: minimize sequential_dma subject to L1 fit; ties broken by total
DMA, then fewer tasks, then smaller L1.

Wave-considered vs non-wave-considered:
  - wave_considered=True (default): rank by sequential_dma and apply the
    max_waves filter. This is the wall-clock proxy described above and is
    what you want for end-to-end perf on a fixed core count.
  - wave_considered=False: rank by total DMA only and skip the max_waves
    filter. Useful when you care about total bytes moved (e.g. comparing
    tile strategies independent of dispatch granularity) or when targeting
    a hardware model where dispatch granularity isn't a wave of fixed
    width. Sequential DMA is still reported as a secondary column.

Usage:
    python hint_l1_tile.py --M 16 --N 5120 --K 17408
    python hint_l1_tile.py --M 16 --N 512  --K 17408
    python hint_l1_tile.py --M 16 --N 256  --K 256
"""

import argparse
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class PinSet:
    """L1 pin choice for A and B. C is always accumulator-resident (L0C), so
    it is not part of the L1 pin sweep."""

    a: bool
    b: bool

    @property
    def label(self) -> str:
        s = ("A" if self.a else "") + ("B" if self.b else "")
        return s if s else "-"


@dataclass
class TileCandidate:
    TM: int
    TN: int
    TK: int
    OM: int
    ON: int
    OK: int
    LM: int
    LN: int
    LK: int
    num_tasks: int
    num_waves: int              # ceil(num_tasks / cores)
    pin: PinSet
    l1_bytes: int
    acc_bytes: int              # C accumulator fragment in L0C (TM*TN*bc)
    dma_total_bytes: int
    dma_per_task_bytes: int     # = dma_total / num_tasks (all tasks equivalent)
    dma_seq_bytes: int          # = dma_per_task * num_waves (wall-clock proxy)
    mac_waste_ratio: float      # m_eff*n_eff*k_eff / (M*N*K) - 1 (compute padding)
    dma_waste_ratio: float      # actual/useful - 1 (per-operand bytes + 512B align)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pow2_up_to(min_val: int, max_val: int) -> list[int]:
    """Powers of 2 in [min_val, max_val], plus max_val. min_val clamped >= 1."""
    floor = max(1, min_val)
    if max_val < floor:
        return [floor]
    out: set[int] = {floor, max_val}
    s = floor
    while s < max_val:
        s *= 2
        if s <= max_val:
            out.add(s)
    return sorted(out)


def _pad_to_multiple(dim: int, multiple: int) -> int:
    """Round dim up to the next multiple of `multiple`."""
    return _ceil_div(dim, multiple) * multiple


def _tile_cands(min_size: int, max_size: int, multiple: int) -> list[int]:
    """Tile size candidates: powers of 2 that are multiples of `multiple`,
    from max(min_size rounded up, multiple) up to max_size. Plus max_size."""
    floor = max(multiple, _pad_to_multiple(min_size, multiple))
    floor = min(floor, max_size)
    out: set[int] = {floor}
    if max_size % multiple == 0:
        out.add(max_size)
    s = floor
    while s < max_size:
        s *= 2
        if s <= max_size and s % multiple == 0:
            out.add(s)
    return sorted(out)


def _tile_l1_min(TM: int, TN: int, TK: int, ba: int, bb: int) -> int:
    """L1 lower bound: single A + B tile, no DB, no pin (cheapest possible).
    C is not in L1 (accumulator-resident), so it does not appear here."""
    return TM * TK * ba + TK * TN * bb


def _acc_used(TM: int, TN: int, bc: int) -> int:
    """L0C accumulator footprint: one TM×TN output fragment (single buffer)."""
    return TM * TN * bc


def _l1_used(
    TM: int, TN: int, TK: int,
    LM: int, LN: int, LK: int,
    ba: int, bb: int,
    pin: PinSet,
) -> int:
    """L1 footprint of A and B only (C lives in the L0C accumulator buffer).

    - Pinned operand: outer-task slice (single buffer; held across the loop).
    - Non-pinned operand: tile, double-buffered (2×) when there's an inner
      loop to overlap; single buffer when the task runs a single iteration.
    """
    db = 2 if (LM * LN * LK > 1) else 1
    a_l1 = (LM * TM) * (LK * TK) * ba if pin.a else db * TM * TK * ba
    b_l1 = (LK * TK) * (LN * TN) * bb if pin.b else db * TK * TN * bb
    return a_l1 + b_l1


def _dma_costs(
    TM: int, TN: int, TK: int,
    OM: int, ON: int, OK: int,
    LM: int, LN: int, LK: int,
    ba: int, bb: int, bc: int,
    pin: PinSet,
    cache_line: int,
    m_eff: int, n_eff: int, k_eff: int,
    M: int, N: int, K: int,
    b_trans: bool,
) -> tuple[int, float]:
    """Return ``(actual_dma, dma_waste)`` across all outer tasks.

    ``actual_dma`` charges each tile its padded footprint, innermost row rounded
    up to ``cache_line`` (sub-cache-line loads pay the full line) — this is the
    DMA cost the ranking uses. The B tile's contiguous row follows the weight
    layout: ``TN*bb`` when ``b_trans`` is False (B=[K,N], N-contiguous), or
    ``TK*bb`` when True (B=[N,K], K-contiguous, the LLM weight layout) — so a
    short K then leaves the B row below the cache line and doubles the B load.

    ``dma_waste = actual_dma / useful_dma - 1`` where ``useful_dma`` costs the
    *same* tiling strategy (identical load/reload counts) but charges each tile
    only its real content: unrounded rows, dims scaled by the unpadded fraction
    (M/m_eff, N/n_eff, K/k_eff). Because each operand spans only its own axes
    (A=[M,K], B spans K&N either way, C=[M,N]), padding M inflates A and C but
    never B, and the cache-line rounding surfaces as alignment waste. Reload
    strategy cancels (same counts in both), isolating padding + alignment."""
    row_a_eff = max(TK * ba, cache_line)
    row_c_eff = max(TN * bc, cache_line)
    a_tile = TM * row_a_eff
    c_tile = TM * row_c_eff
    # B tile: contiguous row depends on the weight layout. Transposed weight
    # (b_trans, B=[N,K]) is K-contiguous → TN rows of TK*bb; else N-contiguous
    # → TK rows of TN*bb. Same TK*TN elements; only which axis meets the line.
    if b_trans:
        b_tile = TN * max(TK * bb, cache_line)
    else:
        b_tile = TK * max(TN * bb, cache_line)

    # Useful (real) per-tile bytes: unrounded rows, each axis scaled to its
    # unpadded fraction. M padding only touches operands that carry M (A, C).
    fM, fN, fK = M / m_eff, N / n_eff, K / k_eff
    a_use = TM * TK * ba * fM * fK
    b_use = TK * TN * bb * fK * fN
    c_use = TM * TN * bc * fM * fN

    num_tasks = OM * ON * OK
    a_loads = LM * LK if pin.a else LM * LN * LK
    b_loads = LN * LK if pin.b else LM * LN * LK
    # C is accumulator-resident: reduced over K in L0C, written to DDR once per
    # TM×TN output fragment (LM*LN per task). Split-K (OK>1) makes that an
    # atomic-add, but the write count is unchanged.
    c_writes = LM * LN

    actual = num_tasks * (a_loads * a_tile + b_loads * b_tile + c_writes * c_tile)
    useful = num_tasks * (a_loads * a_use + b_loads * b_use + c_writes * c_use)
    dma_waste = (actual / useful - 1.0) if useful > 0 else 0.0
    return actual, dma_waste


def hint_l1_tile(
    M: int,
    N: int,
    K: int,
    *,
    bytes_a: int = 2,
    bytes_b: int = 2,
    bytes_c: int = 4,
    l1_available_bytes: int = 512 * 1024,
    acc_buffer_bytes: int = 128 * 1024,
    min_contig_bytes: int = 256,
    m_min: int = 1,
    cores: int = 24,
    tile_multiple: int = 16,
    cache_line_bytes: int = 512,
    max_waves: int = 3,
    wave_considered: bool = True,
    b_trans: bool = False,
) -> list[TileCandidate]:
    """Sweep (TM, TN, TK, OM, ON, OK, pin) and return all feasible candidates.

    A and B occupy L1 (``l1_available_bytes``); C is the matmul accumulator and
    occupies the separate L0C accumulator buffer (``acc_buffer_bytes``). L0C is a
    *soft* wall — the compiler output-tiles an oversized C at a drain cost (see the
    module docstring's Acc-fit note) — but this tool keeps the ``C_acc <=
    acc_buffer_bytes`` skip as the conservative "fits one L0C tile" view.

    When ``wave_considered`` is True (default), candidates are ranked by
    sequential DMA (wall-clock proxy = per_task_dma * ceil(num_tasks / cores))
    and the ``max_waves`` filter is applied; ties broken by total DMA, fewer
    tasks, smaller L1. When False, candidates are ranked by total DMA only
    and the ``max_waves`` filter is skipped — useful when comparing tile
    strategies independent of dispatch granularity.

    Two waste metrics are reported per candidate, for information only — neither
    filters. MAC waste (compute padding vs. the raw M*N*K volume) and DMA waste
    (bytes moved vs. the same strategy with real dims and no cache-line rounding;
    per-operand, so padding M spares B, and it charges the 512 B alignment). Both
    are already priced into the DMA cost that ranks candidates, so filtering on
    them would double-count and, for sub-fragment dims (e.g. M < tile_multiple,
    whose MAC padding is unavoidable and constant across all candidates), could
    prune every otherwise-feasible tiling.

    ``b_trans`` (default False) selects the weight layout for B's DMA row and the
    K/N search floors: False = plain B=[K,N] (N-contiguous); True = B=[N,K],
    K-contiguous (the transposed layout of LLM weight matmuls), where a short K
    tile drops TK*bb below the cache line and doubles the B load.
    """
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"M, N, K must be > 0, got M={M}, N={N}, K={K}")
    if l1_available_bytes <= 0:
        raise ValueError(f"l1_available_bytes must be > 0, got {l1_available_bytes}")
    if acc_buffer_bytes <= 0:
        raise ValueError(f"acc_buffer_bytes must be > 0, got {acc_buffer_bytes}")
    if cores <= 0:
        raise ValueError(f"cores must be > 0, got {cores}")
    if tile_multiple <= 0:
        raise ValueError(f"tile_multiple must be > 0, got {tile_multiple}")
    if cache_line_bytes <= 0:
        raise ValueError(f"cache_line_bytes must be > 0, got {cache_line_bytes}")
    if max_waves <= 0:
        raise ValueError(f"max_waves must be > 0, got {max_waves}")

    # Search floor per dim = the tile size that keeps each operand's *contiguous*
    # axis at least min_contig bytes. A is always K-contiguous ([M,K]); the
    # weight's contiguous axis is N ([K,N]) or K ([N,K]) under b_trans; C is
    # N-contiguous ([M,N]).
    if b_trans:
        perf_min_k = max(1, _ceil_div(min_contig_bytes, bytes_a),
                            _ceil_div(min_contig_bytes, bytes_b))
        perf_min_n = max(1, _ceil_div(min_contig_bytes, bytes_c))
    else:
        perf_min_n = max(1, _ceil_div(min_contig_bytes, bytes_b))
        perf_min_k = max(1, _ceil_div(min_contig_bytes, bytes_a))
    perf_min_m = max(1, m_min)

    # Pad each dim up to the next multiple of `tile_multiple`; tile sizes are
    # constrained to multiples of `tile_multiple` (matmul fragment alignment).
    M_pad = _pad_to_multiple(M, tile_multiple)
    N_pad = _pad_to_multiple(N, tile_multiple)
    K_pad = _pad_to_multiple(K, tile_multiple)

    TM_cands = _tile_cands(perf_min_m, M_pad, tile_multiple)
    TN_cands = _tile_cands(perf_min_n, N_pad, tile_multiple)
    TK_cands = _tile_cands(perf_min_k, K_pad, tile_multiple)

    all_pins = [PinSet(a, b) for a, b in product((False, True), repeat=2)]
    effective_volume = M * N * K
    candidates: list[TileCandidate] = []

    for TM in TM_cands:
        for TN in TN_cands:
            # C accumulator fragment (TM×TN) vs L0C. L0C is a *soft* wall — the
            # compiler (AutoTileMatmulL0) output-tiles an oversized fragment at a
            # per-tile drain this DMA ranker can't price, so we keep a hard skip as the
            # conservative "no compiler output-tiling" view (see module docstring).
            acc = _acc_used(TM, TN, bytes_c)
            if acc > acc_buffer_bytes:
                continue
            for TK in TK_cands:
                if _tile_l1_min(TM, TN, TK, bytes_a, bytes_b) > l1_available_bytes:
                    continue

                # OM × TM × LM ≥ M_pad (with ceil); cap OM at M_pad/TM.
                OM_max = max(1, M_pad // TM)
                ON_max = max(1, N_pad // TN)
                OK_max = max(1, K_pad // TK)
                OM_cands = _pow2_up_to(1, OM_max)
                ON_cands = _pow2_up_to(1, ON_max)
                OK_cands = _pow2_up_to(1, OK_max)

                for OM in OM_cands:
                    LM = _ceil_div(M_pad, OM * TM)
                    m_eff = OM * LM * TM
                    for ON in ON_cands:
                        LN = _ceil_div(N_pad, ON * TN)
                        n_eff = ON * LN * TN
                        for OK in OK_cands:
                            LK = _ceil_div(K_pad, OK * TK)
                            k_eff = OK * LK * TK

                            tiled = m_eff * n_eff * k_eff
                            # MAC waste (compute padding). Reported only; not a
                            # filter — already priced into the DMA cost below.
                            mac_waste = (tiled - effective_volume) / effective_volume

                            num_tasks = OM * ON * OK
                            num_waves = _ceil_div(num_tasks, cores)
                            if wave_considered and num_waves > max_waves:
                                continue

                            for pin in all_pins:
                                l1 = _l1_used(TM, TN, TK, LM, LN, LK,
                                              bytes_a, bytes_b, pin)
                                if l1 > l1_available_bytes:
                                    continue
                                dma_total, dma_waste = _dma_costs(
                                    TM, TN, TK, OM, ON, OK, LM, LN, LK,
                                    bytes_a, bytes_b, bytes_c, pin,
                                    cache_line_bytes,
                                    m_eff, n_eff, k_eff, M, N, K, b_trans)
                                dma_per_task = dma_total // num_tasks
                                dma_seq = dma_per_task * num_waves
                                candidates.append(TileCandidate(
                                    TM=TM, TN=TN, TK=TK,
                                    OM=OM, ON=ON, OK=OK,
                                    LM=LM, LN=LN, LK=LK,
                                    num_tasks=num_tasks,
                                    num_waves=num_waves,
                                    pin=pin,
                                    l1_bytes=l1,
                                    acc_bytes=acc,
                                    dma_total_bytes=dma_total,
                                    dma_per_task_bytes=dma_per_task,
                                    dma_seq_bytes=dma_seq,
                                    mac_waste_ratio=mac_waste,
                                    dma_waste_ratio=dma_waste,
                                ))

    if wave_considered:
        candidates.sort(key=lambda c: (
            c.dma_seq_bytes, c.dma_total_bytes, c.num_tasks, c.l1_bytes,
            c.acc_bytes))
    else:
        candidates.sort(key=lambda c: (
            c.dma_total_bytes, c.num_tasks, c.l1_bytes, c.acc_bytes))
    return candidates


def _fmt_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / (1024**3):.2f} GB"
    if b >= 1024**2:
        return f"{b / (1024**2):.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


def _print(M: int, N: int, K: int, l1: int, acc: int, cores: int,
           cands: list[TileCandidate], top_n: int,
           wave_considered: bool, b_trans: bool) -> None:
    mode = "wave-considered" if wave_considered else "non-wave-considered"
    layout = "B=[N,K] K-contig (b_trans)" if b_trans else "B=[K,N] N-contig"
    rank_metric = "sequential DMA" if wave_considered else "total DMA"
    print(f"\nMatmul (M={M}, N={N}, K={K}), L1 budget {_fmt_bytes(l1)} (A+B), "
          f"Acc budget {_fmt_bytes(acc)} (C), per core, {cores} cores "
          f"[{mode}; {layout}]")
    print(
        "  Notation: each axis decomposes as dim = O * L * T:\n"
        "    T (Tile)  = L1-resident fragment per load/store\n"
        "    L (Loop)  = sequential iterations inside one core for one outer task\n"
        "    O (Outer) = parallel outer tasks dispatched across cores\n"
        "  E.g. for the N axis: N ≈ ON * LN * TN. Tasks = OM*ON*OK; "
        "Waves = ceil(Tasks/cores).\n"
        "  Pin = subset of {A,B} held resident in L1 across the loop "
        "(saves DMA at L1 cost).\n"
        "  C is always accumulator-resident (L0C, never L1); 'Acc' = TM*TN*bc "
        "fragment, must fit the Acc budget.\n"
        "  DMAw = per-operand DMA waste (padding + 512 B cache-line align; M "
        "padding spares B); MACw = compute-padding waste. Both informational."
    )
    shown = cands[:top_n]
    print(
        f"\n{len(cands)} feasible (tile, outer, pin) combo(s); "
        f"showing top {len(shown)} by ascending {rank_metric}:\n"
    )
    header = (
        f"  {'TM':>4} {'TN':>4} {'TK':>6} "
        f"{'OM':>3} {'ON':>4} {'OK':>4} "
        f"{'LM':>3} {'LN':>4} {'LK':>4} "
        f"{'Tasks':>6} {'Wv':>3} {'Pin':>3} "
        f"{'L1':>10} {'Acc':>10} {'Seq DMA':>11} {'Tot DMA':>11} "
        f"{'DMAw':>6} {'MACw':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in shown:
        print(
            f"  {c.TM:>4} {c.TN:>4} {c.TK:>6} "
            f"{c.OM:>3} {c.ON:>4} {c.OK:>4} "
            f"{c.LM:>3} {c.LN:>4} {c.LK:>4} "
            f"{c.num_tasks:>6} {c.num_waves:>3} {c.pin.label:>3} "
            f"{_fmt_bytes(c.l1_bytes):>10} "
            f"{_fmt_bytes(c.acc_bytes):>10} "
            f"{_fmt_bytes(c.dma_seq_bytes):>11} "
            f"{_fmt_bytes(c.dma_total_bytes):>11} "
            f"{c.dma_waste_ratio * 100:>5.1f}% "
            f"{c.mac_waste_ratio * 100:>5.1f}%"
        )
    print()
    if shown:
        b = shown[0]
        print(f"  Recommended (smallest {rank_metric}): "
              f"tile=({b.TM}, {b.TN}, {b.TK}), "
              f"outer=({b.OM}, {b.ON}, {b.OK}), "
              f"loop=({b.LM}, {b.LN}, {b.LK}), pin={b.pin.label} (C always in Acc)")
        print(f"    Tasks: {b.num_tasks} ({b.num_waves} wave(s) on {cores} cores), "
              f"L1: {_fmt_bytes(b.l1_bytes)}, Acc: {_fmt_bytes(b.acc_bytes)}")
        print(f"    Per-task DMA: {_fmt_bytes(b.dma_per_task_bytes)}, "
              f"Sequential DMA: {_fmt_bytes(b.dma_seq_bytes)}, "
              f"Total DMA: {_fmt_bytes(b.dma_total_bytes)}")
        print(f"    DMA waste: {b.dma_waste_ratio * 100:.1f}% "
              f"(bytes; the bandwidth number), "
              f"MAC waste: {b.mac_waste_ratio * 100:.1f}% (compute)")
    else:
        print(f"  No configuration fits in L1 budget {_fmt_bytes(l1)} / "
              f"Acc budget {_fmt_bytes(acc)}.")
    print()


def _main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--M", type=int, required=True, help="Matmul M dim")
    p.add_argument("--N", type=int, required=True, help="Matmul N dim")
    p.add_argument("--K", type=int, required=True, help="Matmul K dim")
    p.add_argument("--bytes-a", type=int, default=2, help="A elem bytes (default 2 = BF16)")
    p.add_argument("--bytes-b", type=int, default=2, help="B elem bytes (default 2 = BF16)")
    p.add_argument("--bytes-c", type=int, default=4,
                   help="C accumulator elem bytes (default 4 = FP32 cube accumulator)")
    p.add_argument(
        "--l1-bytes",
        type=int,
        default=512 * 1024,
        help="L1 physical budget per core in bytes (default 512 KB), holding A "
             "and B only; double-buffer cost is modeled per non-pinned operand",
    )
    p.add_argument(
        "--acc-bytes",
        type=int,
        default=128 * 1024,
        help="L0C accumulator buffer per core in bytes (default 128 KB on 910B, "
             "256 KB on a5/950). C never uses L1. L0C is a *soft* wall — the compiler "
             "output-tiles an oversized fragment at a drain cost; this filter is the "
             "conservative 'fits one L0C tile' boundary.",
    )
    p.add_argument(
        "--min-contig",
        type=int,
        default=256,
        help="Min contiguous bytes per inner-most tile row — search floor for "
             "tile width (default 256). Tiles below this are not enumerated.",
    )
    p.add_argument(
        "--cache-line",
        type=int,
        default=512,
        help="Hardware cache-line size used to round DMA row strides "
             "(default 512 B on Ascend 910B L2). Sub-cache-line loads are "
             "allowed but charged the full line in the DMA cost.",
    )
    p.add_argument(
        "--m-min",
        type=int,
        default=1,
        help="Min M tile size (M has no contig constraint, default 1)",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=24,
        help="Number of parallel AIC cores; tasks dispatch in waves of this many (default 24)",
    )
    p.add_argument(
        "--max-waves",
        type=int,
        default=3,
        help="Discard configs with more than this many dispatch waves "
             "(default 3). Caps per-task dispatch overhead. "
             "Ignored when --no-wave-considered is set.",
    )
    p.add_argument(
        "--no-wave-considered",
        dest="wave_considered",
        action="store_false",
        help="Rank candidates by total DMA only and skip the --max-waves "
             "filter. Default is wave-considered: rank by sequential DMA "
             "(per_task * ceil(tasks/cores)) and apply --max-waves.",
    )
    p.set_defaults(wave_considered=True)
    p.add_argument(
        "--b-trans",
        dest="b_trans",
        action="store_true",
        help="Model the weight B as transposed / K-contiguous ([N,K], the "
             "b_trans layout used by LLM weight matmuls): B's DMA row is TK*bb, "
             "so a sub-cache-line K wastes the line and doubles the B load. "
             "Default off (plain row-major B=[K,N], N-contiguous row TN*bb).",
    )
    p.set_defaults(b_trans=False)
    p.add_argument(
        "--tile-multiple",
        type=int,
        default=16,
        help="Tile sizes must be multiples of this (matmul fragment alignment, "
             "default 16); dims are padded up with waste accounted",
    )
    p.add_argument(
        "--top",
        type=int,
        default=12,
        help="Max rows to print, sorted by ascending sequential DMA (default 12)",
    )
    args = p.parse_args()

    cands = hint_l1_tile(
        args.M,
        args.N,
        args.K,
        bytes_a=args.bytes_a,
        bytes_b=args.bytes_b,
        bytes_c=args.bytes_c,
        l1_available_bytes=args.l1_bytes,
        acc_buffer_bytes=args.acc_bytes,
        min_contig_bytes=args.min_contig,
        m_min=args.m_min,
        cores=args.cores,
        tile_multiple=args.tile_multiple,
        cache_line_bytes=args.cache_line,
        max_waves=args.max_waves,
        wave_considered=args.wave_considered,
        b_trans=args.b_trans,
    )
    _print(args.M, args.N, args.K, args.l1_bytes, args.acc_bytes, args.cores,
           cands, args.top, args.wave_considered, args.b_trans)


if __name__ == "__main__":
    _main()
