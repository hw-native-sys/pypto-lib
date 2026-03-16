# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

r"""
# Batch Hash Lookup — no per-lane sequential loop

This example demonstrates a SIMD-friendly lookup path for `[1, 32]` tiles.

- Hash function: `h = (key * GOLDEN_PRIME) & (TABLE_BUCKET_COUNT - 1)`.
- Execution model: all 32 lanes are computed together with tile ops.
- No `for j in pl.range(32)` loop is used in the program.

`hash_pool` is organized per `table_i` into two dense planes:

1. key plane  : `[bucket][table_j]`
2. value plane: `[bucket][table_j]`

During lookup, the kernel computes `h_tile` once, then scans all buckets
(`pl.range(TABLE_BUCKET_COUNT)`) and uses `cmps/cmp/and_/sel` tile masks to
pick the matched value per lane. Probe rounds use ``pl.break_()`` for early
exit when no lane has active lookups (all found).
"""

import os

import pypto.language as pl


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

BATCH_NUM = 1024
NUM_TABLES_I = 64
NUM_TABLES_J = 32

TABLE_BUCKET_COUNT = 64         # primary buckets per table (power-of-2)
GOLDEN_PRIME = 2654435761       # Knuth's multiplicative hash constant
MAX_PROBE_ROUNDS = 8
PROBE_STRIDE = 2246822519       # odd stride for probe-round remapping

HASH_PLANE_ROWS = TABLE_BUCKET_COUNT
VALUE_PLANE_ROWS = TABLE_BUCKET_COUNT
POOL_ROWS = HASH_PLANE_ROWS + VALUE_PLANE_ROWS
POOL_SIZE = NUM_TABLES_I * POOL_ROWS * NUM_TABLES_J
NUM_INSERT_PER_TABLE = 1        # one key per (table_i, table_j) for direct bucket lookup

# ══════════════════════════════════════════════════════════════════════════════
#  PyPTO Program
# ══════════════════════════════════════════════════════════════════════════════

def build_batch_hash_lookup_program(
    batch_num: int = BATCH_NUM,
    num_tables_i: int = NUM_TABLES_I,
    num_tables_j: int = NUM_TABLES_J,
    pool_size: int = POOL_SIZE,
):
    """Build the batch hash lookup ``@pl.program``.

    Returns a PyPTO program class with a single ``batch_hash_lookup`` function
    that performs tile-parallel hash lookups across all (batch, table_i, table_j)
    keys.
    """
    NTJ = num_tables_j

    @pl.program
    class BatchHashLookup:

        @pl.function(type=pl.FunctionType.Opaque)
        def batch_hash_lookup(
            self,
            search_key: pl.Tensor[[batch_num, num_tables_i, num_tables_j], pl.INT32],
            hash_table_size: pl.Tensor[[num_tables_i, num_tables_j], pl.INT32],
            hash_base_ptr: pl.Tensor[[num_tables_i, num_tables_j], pl.INT32],
            hash_pool: pl.Tensor[[num_tables_i, POOL_ROWS, num_tables_j], pl.INT32],
            value_ptr_out: pl.Tensor[[batch_num, num_tables_i, num_tables_j], pl.INT32],
        ) -> pl.Tensor[[batch_num, num_tables_i, num_tables_j], pl.INT32]:
            """Batch hash lookup with outer probe rounds and inner tile batching.

            For each ``(b, ti)`` work-item the 32 ``table_j`` lookups form a
            natural tile vector.  Three phases execute per work-item:

            1. **Hash seed** — compute all-lane mixed hash once.
            2. **Probe rounds** — sequential rounds outside, tile batch inside.
            3. **Write** — single tile-level assemble of [1, 32] results.
            """


            # Initialize output tiles once so later probe rounds can read/write
            # them as state without relying on external zero-init behavior.
            for b in pl.parallel(0, batch_num, 32):
                with pl.incore():
                    for ti in pl.parallel(0, num_tables_i, 32):
                        zero_src: pl.Tensor[[1, NTJ], pl.INT32] = pl.view(
                            search_key, [1, NTJ], [b, ti, 0]
                        )
                        zero_tile = pl.mul(zero_src, 0)
                        value_ptr_out = pl.assemble(value_ptr_out, zero_tile, [b, ti, 0])


            # Probe rounds are the outermost loop; use pl.break for early exit
            # when no lane has active lookups (all found).
            for probe in pl.range(MAX_PROBE_ROUNDS):
                round_has_active = 0
                with pl.incore():
                    for b in pl.parallel(0, batch_num, 32):
                        for ti in pl.parallel(0, num_tables_i, 32):
                            keys_tile: pl.Tensor[[1, NTJ], pl.INT32] = pl.view(
                                search_key, [1, NTJ], [b, ti, 0]
                            )
                            mixed = pl.mul(keys_tile, GOLDEN_PRIME)
                            h_probe = pl.ands(
                                pl.add(mixed, probe * PROBE_STRIDE),
                                TABLE_BUCKET_COUNT - 1,
                            )

                            cand_key = pl.mul(keys_tile, 0)
                            cand_val = pl.mul(keys_tile, 0)

                            for bucket in pl.range(TABLE_BUCKET_COUNT):
                                bucket_mask = pl.cmps(h_probe, bucket, cmp_type=0)
                                bucket_keys = pl.view(
                                    hash_pool, [1, NTJ], [ti, bucket, 0]
                                )
                                bucket_vals = pl.view(
                                    hash_pool,
                                    [1, NTJ],
                                    [ti, HASH_PLANE_ROWS + bucket, 0],
                                )
                                cand_key = pl.sel(bucket_mask, bucket_keys, cand_key)
                                cand_val = pl.sel(bucket_mask, bucket_vals, cand_val)

                            result_prev = pl.view(value_ptr_out, [1, NTJ], [b, ti, 0])
                            active_mask = pl.cmps(result_prev, 0, cmp_type=0)
                            active_count = pl.row_sum(active_mask)
                            active_count_s = pl.tensor.read(active_count, [0, 0])
                            if active_count_s != 0:
                                round_has_active = 1
                            key_match = pl.cmp(cand_key, keys_tile, cmp_type=0)
                            hit_mask = pl.and_(active_mask, key_match)
                            result_next = pl.sel(hit_mask, cand_val, result_prev)
                            value_ptr_out = pl.assemble(
                                value_ptr_out, result_next, [b, ti, 0]
                            )
                if round_has_active == 0:
                    pl.break_()

            return value_ptr_out

    return BatchHashLookup


# ══════════════════════════════════════════════════════════════════════════════
#  Golden reference (Python / NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def golden(tensors, params=None):
    """Reference hash lookup implementation for correctness verification."""
    import numpy as np
    import torch  # type: ignore[import]

    sk = tensors["search_key"].numpy()
    hp = tensors["hash_pool"].numpy()
    out = np.zeros_like(sk, dtype=np.int32)

    bsz, ni, nj = sk.shape
    mixed = sk.astype(np.int64) * np.int64(GOLDEN_PRIME)
    found = np.zeros_like(sk, dtype=bool)

    i_idx = np.broadcast_to(
        np.arange(ni, dtype=np.int64)[None, :, None], (bsz, ni, nj)
    )
    j_idx = np.broadcast_to(
        np.arange(nj, dtype=np.int64)[None, None, :], (bsz, ni, nj)
    )

    for probe in range(MAX_PROBE_ROUNDS):
        h = (mixed + np.int64(probe) * np.int64(PROBE_STRIDE)) & np.int64(
            TABLE_BUCKET_COUNT - 1
        )
        h = h.astype(np.int64)
        cand_key = hp[i_idx, h, j_idx]
        cand_val = hp[i_idx, HASH_PLANE_ROWS + h, j_idx]
        new_hit = (~found) & (cand_key == sk)
        out[new_hit] = cand_val[new_hit]
        found |= new_hit
        if found.all():
            break

    tensors["value_ptr_out"][:] = torch.from_numpy(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Test data & TensorSpec builder
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor_specs(
    batch_num: int = BATCH_NUM,
    num_tables_i: int = NUM_TABLES_I,
    num_tables_j: int = NUM_TABLES_J,
    pool_size: int = POOL_SIZE,
):
    """Build ``TensorSpec`` list with SIMD-friendly bucket-plane hash tables."""
    import numpy as np
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    rng = np.random.default_rng(42)

    hp = np.zeros((num_tables_i, POOL_ROWS, num_tables_j), dtype=np.int32)
    ht_sz = np.full((num_tables_i, num_tables_j), TABLE_BUCKET_COUNT, dtype=np.int32)
    ht_bp = np.zeros((num_tables_i, num_tables_j), dtype=np.int32)

    ht_bp[:, :] = 0

    # One inserted key/value per (table_i, table_j), generated in parallel by NumPy.
    inserted_keys = rng.integers(
        1, 2**30, size=(num_tables_i, num_tables_j), dtype=np.int32
    )
    inserted_vals = rng.integers(
        1, 2**30, size=(num_tables_i, num_tables_j), dtype=np.int32
    )
    probe_round = rng.integers(
        0, MAX_PROBE_ROUNDS, size=(num_tables_i, num_tables_j), dtype=np.int32
    )
    h = (
        inserted_keys.astype(np.int64) * np.int64(GOLDEN_PRIME)
        + probe_round.astype(np.int64) * np.int64(PROBE_STRIDE)
    ) & np.int64(TABLE_BUCKET_COUNT - 1)
    h = h.astype(np.int64)
    ii, jj = np.indices((num_tables_i, num_tables_j), dtype=np.int64)
    hp[ii, h, jj] = inserted_keys
    hp[ii, HASH_PLANE_ROWS + h, jj] = inserted_vals

    # Search keys: ~50 % matching, ~50 % non-matching
    sk = rng.integers(
        2**29, 2**30, size=(batch_num, num_tables_i, num_tables_j), dtype=np.int32
    )
    match_mask = rng.random((batch_num, num_tables_i, num_tables_j)) < 0.5
    keys_broadcast = np.broadcast_to(
        inserted_keys[np.newaxis, :, :], (batch_num, num_tables_i, num_tables_j)
    )
    sk = np.where(match_mask, keys_broadcast, sk).astype(np.int32)

    return [
        TensorSpec(
            "search_key",
            [batch_num, num_tables_i, num_tables_j],
            torch.int32,
            init_value=torch.from_numpy(sk.copy()),
        ),
        TensorSpec(
            "hash_table_size",
            [num_tables_i, num_tables_j],
            torch.int32,
            init_value=torch.from_numpy(ht_sz.copy()),
        ),
        TensorSpec(
            "hash_base_ptr",
            [num_tables_i, num_tables_j],
            torch.int32,
            init_value=torch.from_numpy(ht_bp.copy()),
        ),
        TensorSpec(
            "hash_pool",
            [num_tables_i, POOL_ROWS, num_tables_j],
            torch.int32,
            init_value=torch.from_numpy(hp.copy()),
        ),
        TensorSpec(
            "value_ptr_out",
            [batch_num, num_tables_i, num_tables_j],
            torch.int32,
            is_output=True,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Compile & run
# ══════════════════════════════════════════════════════════════════════════════

def compile_and_run(
    batch_num: int = BATCH_NUM,
    num_tables_i: int = NUM_TABLES_I,
    num_tables_j: int = NUM_TABLES_J,
    pool_size: int = POOL_SIZE,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    """Compile (and optionally run) the batch hash lookup program."""
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_batch_hash_lookup_program(
        batch_num=batch_num,
        num_tables_i=num_tables_i,
        num_tables_j=num_tables_j,
        pool_size=pool_size,
    )
    tensor_specs = build_tensor_specs(
        batch_num=batch_num,
        num_tables_i=num_tables_i,
        num_tables_j=num_tables_j,
        pool_size=pool_size,
    )

    if work_dir is None:
        work_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "batch_hash_lookup_dump")
        )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=0,
            atol=0,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Generated kernels/orchestration:", work_dir)
        return result
    if result.passed:
        print("  Generated kernels/orchestration:", work_dir)
    else:
        print(f"Result: {result.error}")
        print("  Pass dumps may still have been written to:", work_dir)
    return result


if __name__ == "__main__":
    compile_and_run()
