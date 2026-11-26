# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Fuse HC pre-mix, projection and top-6 routing on A5 Cube groups."""

import os
from functools import lru_cache
from pathlib import Path

import pypto.language as pl

if __name__ == "__main__":
    import config

    config.set_moe_tokens(384)

from gate import route_full_width_dynamic
from hc import hc_pre_moe_tiled, hc_sinkhorn


MIN_REUSE_ROWS = 160
MAX_REUSE_ROWS = 1024
LOGIT_ROWS = pl.dynamic("HC_ROUTER_LOGIT_ROWS")


def select_hc_router(experts, topk):
    """Compile a live-prefix dispatch for the opted-in full-width router."""
    if os.getenv("PYPTO_V41_HC_ROUTER_REUSE") == "1" and experts == 384 and topk == 6:
        return make_hc_router()
    return None


def live_prefix_contract(experts, topk, runtime_dir):
    """The validator must honor the compiled output contract even on artifact replay."""
    if runtime_dir is None:
        return select_hc_router(experts, topk) is not None
    return any("hc_router_live_clear" in f.read_text() for f in Path(runtime_dir).rglob("kernel_config.py"))


def compare_live_ranks(compare, counts):
    """Apply the same numerical tolerance independently to each rank's valid prefix."""
    def compare_prefix(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.shape[0] != len(counts):
            return False, "unexpected output shape for the rank-local HC prefixes"
        actual_counts = kwargs.get("inputs", {}).get("num_tokens")
        if actual_counts is not None and actual_counts.reshape(-1).tolist() != counts.reshape(-1).tolist():
            return False, "num_tokens on replay differs from the per-rank fixture"
        for rank, count in enumerate(counts.tolist()):
            if not 0 <= count <= actual.shape[1]:
                return False, f"invalid active row count {count} on rank {rank}"
            if count:
                ok, message = compare(actual[rank, :count], expected[rank, :count], **kwargs)
                if not ok:
                    return False, f"rank {rank}, first {count} rows: {message}"
        return True, ""
    return compare_prefix


@lru_cache(None)
def _make_cube_router():
    kernels = Path(__file__).parent / "kernels"
    source = kernels / "hc_router.cpp"

    @pl.jit.extern(
        core_type="mixed", aic_source=source, aiv_source=source, dual_aiv_dispatch=True, include_dirs=(kernels,)
    )
    def _hc_gate_reuse(
        x: pl.Tensor,
        incoming: pl.Tensor,
        fn: pl.Tensor,
        scale: pl.Tensor,
        bias: pl.Tensor,
        gamma: pl.Tensor,
        raw: pl.Out[pl.Tensor],
        norm: pl.Out[pl.Tensor],
        pre: pl.Out[pl.Tensor],
        post: pl.Out[pl.Tensor],
        res: pl.Out[pl.Tensor],
        gate_w: pl.Tensor,
        gate_b: pl.Tensor,
        projection: pl.Out[pl.Tensor],
        logits: pl.Out[pl.Tensor[[LOGIT_ROWS, 384], pl.FP32]],
        indices: pl.Out[pl.Tensor],
        weights: pl.Out[pl.Tensor],
        active: pl.Scalar[pl.INT32],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        return raw, norm, pre, post, res, projection, logits, indices, weights

    @pl.jit.inline(auto_scope=False)
    def hc_gate(
        x: pl.Tensor,
        incoming: pl.Tensor,
        fn: pl.Tensor,
        scale: pl.Tensor,
        bias: pl.Tensor,
        gamma: pl.Tensor,
        raw: pl.Tensor,
        norm: pl.Tensor,
        pre: pl.Tensor,
        post: pl.Tensor,
        res: pl.Tensor,
        gate_w: pl.Tensor,
        gate_b: pl.Tensor,
        indices: pl.Tensor,
        weights: pl.Tensor,
        active: pl.Scalar[pl.INT32],
        clusters: pl.Scalar[pl.INDEX],
        ready: pl.Array[1, pl.TASK_ID],
    ):
        rows = pl.tensor.dim(x, 0)
        waves = (rows + 511) // 512
        groups = waves * 8
        proj_rows = 4 * rows + groups * 16
        proj = pl.create_tensor([proj_rows, 32], dtype=pl.FP32)
        logits = pl.create_tensor([rows, 384], dtype=pl.FP32)
        with pl.spmd(waves, name_hint="hc_router_live_clear") as init:
            first = 4 * rows + pl.tile.get_block_idx() * 128
            proj[first:first + 128, :] = pl.full([128, 32], value=0.0, dtype=pl.FP32)
        workers = pl.min(groups * 4, clusters // 4 * 4)
        with pl.spmd(workers, sync_start=True, name_hint="hc_router", deps=[init]) as done:
            raw, norm, pre, post, res, proj, logits, indices, weights = _hc_gate_reuse(
                x, incoming, fn, scale, bias, gamma, raw, norm, pre, post, res,
                gate_w, gate_b, proj, logits, indices, weights, active,
            )
        ready[0] = done
    return hc_gate


@pl.jit.inline(auto_scope=False)
def upstream_hc(
    tile: pl.constexpr, x: pl.Tensor, incoming: pl.Tensor, fn: pl.Tensor, scale: pl.Tensor,
    bias: pl.Tensor, gamma: pl.Tensor, raw: pl.Tensor, norm: pl.Tensor, pre: pl.Tensor,
    post: pl.Tensor, res: pl.Tensor, active: pl.Scalar[pl.INT32], ready: pl.Array[1, pl.TASK_ID],
):
    inv_rms, partials = hc_pre_moe_tiled(x, incoming, gamma, norm, raw, active, fn, tile)
    ready[0] = hc_sinkhorn(inv_rms, partials, scale, bias, pre, post, res)


@lru_cache(None)
def make_hc_router():
    """Compute the live prefix; only a native tile may extend into available padding."""
    cube_router = _make_cube_router()

    @pl.jit.inline(auto_scope=False)
    def hc_gate(
        x: pl.Tensor,
        incoming: pl.Tensor,
        fn: pl.Tensor,
        scale: pl.Tensor,
        bias: pl.Tensor,
        gamma: pl.Tensor,
        raw: pl.Tensor,
        norm: pl.Tensor,
        pre: pl.Tensor,
        post: pl.Tensor,
        res: pl.Tensor,
        gate_w: pl.Tensor,
        gate_b: pl.Tensor,
        indices: pl.Tensor,
        weights: pl.Tensor,
        active: pl.Scalar[pl.INT32],
        clusters: pl.Scalar[pl.INDEX],
        ready: pl.Array[1, pl.TASK_ID],
    ):
        capacity = pl.tensor.dim(x, 0)
        live = pl.max(0, pl.min(pl.cast(active, pl.INDEX), capacity))
        aligned = ((live + 15) // 16) * 16
        use_cube = live >= MIN_REUSE_ROWS and live <= MAX_REUSE_ROWS and clusters >= 4
        if live == 0:
            ready[0] = pl.system.task_dummy(deps=[])
        else:
            work = live
            if use_cube:
                work = aligned
                if aligned > capacity:
                    work = live // 16 * 16
            # Views share storage; rounding a tile never extends the underlying allocation.
            hx = pl.slice(x, [work, 4, 5120], [0, 0, 0])
            mix = pl.slice(incoming, [work, 4], [0, 0])
            rx = pl.slice(raw, [work, 5120], [0, 0])
            nx = pl.slice(norm, [work, 5120], [0, 0])
            p = pl.slice(pre, [work, 4], [0, 0])
            q = pl.slice(post, [work, 4], [0, 0])
            r = pl.slice(res, [work, 4, 4], [0, 0, 0])
            ids = pl.slice(indices, [work, 6], [0, 0])
            ws = pl.slice(weights, [work, 6], [0, 0])
            count = pl.cast(pl.min(live, work), pl.INT32)
            if use_cube:
                cube_router(hx, mix, fn, scale, bias, gamma, rx, nx, p, q, r,
                            gate_w, gate_b, ids, ws, count, clusters, ready)
                if work < live:
                    # Tight storage: complete native tiles keep the same policy; handle just the short tail.
                    tail = live - work
                    tx = pl.slice(x, [tail, 4, 5120], [work, 0, 0])
                    tm = pl.slice(incoming, [tail, 4], [work, 0])
                    tr = pl.slice(raw, [tail, 5120], [work, 0])
                    tn = pl.slice(norm, [tail, 5120], [work, 0])
                    tp = pl.slice(pre, [tail, 4], [work, 0])
                    tq = pl.slice(post, [tail, 4], [work, 0])
                    tc = pl.slice(res, [tail, 4, 4], [work, 0, 0])
                    ti = pl.slice(indices, [tail, 6], [work, 0])
                    tw = pl.slice(weights, [tail, 6], [work, 0])
                    bias_begin = 0
                    bias_len = 384
                    if pl.tensor.dim(gate_b, 0) > 384:
                        bias_begin = work * 384
                        bias_len = tail * 384
                    tb = pl.slice(gate_b, [bias_len], [bias_begin])
                    tail_count = pl.cast(tail, pl.INT32)
                    tail_ready = pl.array.create(1, pl.TASK_ID)
                    upstream_hc(16, tx, tm, fn, scale, bias, gamma, tr, tn, tp, tq, tc, tail_count, tail_ready)
                    route_full_width_dynamic(tn, gate_w, tb, tail_count, ti, tw)
                    ready[0] = pl.system.task_dummy(deps=[ready[0], tail_ready[0]])
            else:
                if live <= 128:
                    upstream_hc(16, hx, mix, fn, scale, bias, gamma, rx, nx, p, q, r, count, ready)
                elif live <= 256:
                    upstream_hc(32, hx, mix, fn, scale, bias, gamma, rx, nx, p, q, r, count, ready)
                else:
                    upstream_hc(64, hx, mix, fn, scale, bias, gamma, rx, nx, p, q, r, count, ready)
                route_full_width_dynamic(nx, gate_w, gate_b, count, ids, ws)
    return hc_gate


if __name__ == "__main__":
    import argparse
    import json
    import time

    import torch

    from config import T_DYN
    from gate import golden_gate_reference
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from hc import D, HC_DIM, MIX_HC, golden_hc_collapse, golden_hc_head_norm, golden_hc_mixes, make_hc_inputs


    BIAS_DYN = pl.dynamic("router_check_bias")
    E = 384


    def make_check():
        hc_gate = select_hc_router(E, 6)
        if hc_gate is None:
            raise AssertionError("production selector did not enable the full-width router")

        @pl.jit
        def check_hc_pre(
            x: pl.Tensor[[T_DYN, 4, D], pl.FP32],
            incoming: pl.Tensor[[T_DYN, 4], pl.FP32],
            fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            scale: pl.Tensor[[3], pl.FP32],
            bias: pl.Tensor[[MIX_HC], pl.FP32],
            gamma: pl.Tensor[[D], pl.BF16],
            raw: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
            norm: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
            pre: pl.Out[pl.Tensor[[T_DYN, 4], pl.FP32]],
            post: pl.Out[pl.Tensor[[T_DYN, 4], pl.FP32]],
            residual: pl.Out[pl.Tensor[[T_DYN, 4, 4], pl.FP32]],
            gw: pl.Tensor[[E, D], pl.BF16],
            gb: pl.Tensor[[BIAS_DYN], pl.FP32],
            ids: pl.Out[pl.Tensor[[T_DYN, 6], pl.INT32]],
            ws: pl.Out[pl.Tensor[[T_DYN, 6], pl.FP32]],
            active: pl.Scalar[pl.INT32],
            cluster_limit: pl.Scalar[pl.INT32],
        ):
            x.bind_dynamic(0, T_DYN)
            gb.bind_dynamic(0, BIAS_DYN)
            clusters = pl.min(pl.system.available_cluster_count(), cluster_limit)
            done = pl.array.create(1, pl.TASK_ID)
            hc_gate(x, incoming, fn, scale, bias, gamma, raw, norm, pre, post, residual, gw, gb, ids, ws, active, clusters, done)
            return norm
        return check_hc_pre


    def reference(t):
        live = int(t["active"])
        if live == 0:
            return
        x, mix = t["x"][:live], t["incoming"][:live]
        t["raw"][:live] = golden_hc_collapse(x, mix).bfloat16()
        norm = golden_hc_head_norm(x, mix, t["gamma"])
        t["norm"][:live] = norm
        wt, idx = golden_gate_reference(norm, t["gw"], t["gb"])
        t["ids"][:live], t["ws"][:live] = idx, wt
        for key, value in zip(("pre", "post", "residual"), golden_hc_mixes(x, t["fn"], t["scale"], t["bias"])):
            t[key][:live].copy_(value)


    def check_dispatched_path(artifact, before, *, rows, active, cluster_limit):
        # Check the dispatched task IDs, since both branches produce the same numerical outputs.
        captures = [p for p in Path(artifact).rglob("deps.json") if p.stat().st_mtime_ns >= before]
        if len(captures) != 1:
            raise AssertionError(f"expected one fresh device task graph, got {captures}")
        graph = captures[0]
        names = json.loads(next(graph.parent.glob("name_map_*.json")).read_text())["callable_id_to_name"]
        tasks = json.loads(graph.read_text())["tasks"]
        dispatched = {names[str(i)] for task in tasks for i in task["kernel_ids"] if i >= 0}
        native = "_hc_gate_reuse_aic" in dispatched and "_hc_gate_reuse_aiv" in dispatched
        upstream = any(name.startswith("gate_project") for name in dispatched)
        aligned = (active + 15) // 16 * 16
        expect_native = 160 <= active <= 1024 and cluster_limit >= 4
        work = (aligned if aligned <= rows else active // 16 * 16) if expect_native else active
        has_tail = expect_native and work < active
        projections = [t for t in tasks if any(
            names[str(i)].startswith(("hc_linear", "_hc_gate_reuse_aic")) for i in t["kernel_ids"] if i >= 0
        )]
        extents = [t["args"][0]["shape"] for t in projections]
        expected_extents = ([[work, 4, D]] if expect_native else [[work, HC_DIM]]) if active else []
        if has_tail:
            expected_extents.append([active - work, HC_DIM])
        if sorted(extents) != sorted(expected_extents):
            raise AssertionError(f"HC input views: expected {expected_extents}, dispatched {extents}")
        if native != expect_native or upstream != (has_tail or (active > 0 and not expect_native)):
            raise AssertionError(f"incorrect HC/router dispatch: {dispatched=}, {expect_native=}")


    def make_specs(rows, active, *, broadcast_bias):
        ts = make_hc_inputs(rows, seed=71, case="normal")
        gen = torch.Generator().manual_seed(386)
        gw = (torch.randn(E, D, generator=gen) / D**0.5).bfloat16()
        # Rotate six well-separated winners across all four gate panels and rows.
        bias_rows = 1 if broadcast_bias else rows
        winners = (torch.arange(bias_rows)[:, None] * 17 + torch.arange(6) * 61) % E
        gb = torch.zeros(bias_rows, E)
        gb.scatter_(1, winners, torch.arange(6, 0, -1, dtype=torch.float32).expand(bias_rows, -1))
        specs = [
            TensorSpec(dst, list(ts[src].shape), ts[src].dtype, init_value=ts[src])
            for dst, src in zip(
                ("x", "incoming", "fn", "scale", "bias", "gamma"),
                ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight"),
            )
        ]
        for key, shape, dt in [
            ("raw", [D], torch.bfloat16),
            ("norm", [D], torch.bfloat16),
            ("pre", [4], torch.float32),
            ("post", [4], torch.float32),
            ("residual", [4, 4], torch.float32),
        ]:
            specs.append(TensorSpec(key, [rows, *shape], dt))
        specs += [
            TensorSpec("gw", [E, D], torch.bfloat16, init_value=gw),
            TensorSpec("gb", [bias_rows * E], torch.float32, init_value=gb.flatten()),
        ]
        specs += [
            TensorSpec("ids", [rows, 6], torch.int32),
            TensorSpec("ws", [rows, 6], torch.float32),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        return specs


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
        parser.add_argument("-d", "--device", type=int, default=0)
        parser.add_argument("--compile-only", action="store_true")
        parser.add_argument("--runtime-dir")
        args = parser.parse_args()
        # This standalone entry opts in before compiling, exactly as the model callers do.
        os.environ["PYPTO_V41_HC_ROUTER_REUSE"] = "1"
        for experts, topk in ((24, 6), (E, 4)):
            if select_hc_router(experts, topk) is not None:
                raise AssertionError(f"unexpected native router for {experts=}, {topk=}")
        check_hc_pre = make_check()
        artifact = args.runtime_dir
        # Storage rows, live rows, cluster limit, broadcast bias.
        cases = (
            (384, 72, 32, False), (384, 159, 32, False), (384, 160, 32, True),
            (384, 161, 32, False), (161, 161, 32, False), (384, 160, 3, False),
            (384, 0, 32, False),
        )
        for rows, active, cluster_limit, broadcast_bias in cases:
            specs = make_specs(rows, active, broadcast_bias=broadcast_bias)
            exact = ratio_allclose(atol=0, rtol=0, max_error_ratio=0, valid_rows=active)
            coef = ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0, valid_rows=active)
            # Use a single artifact on either side of the four-cluster boundary.
            before = time.time_ns()
            result = run(
                fn=check_hc_pre,
                specs=[*specs, ScalarSpec("cluster_limit", torch.int32, cluster_limit, compile_runtime=True)],
                golden_fn=reference,
                compare_fn={"raw": exact, "norm": ratio_allclose(atol=1e-4, rtol=1 / 128, max_error_ratio=0, valid_rows=active),
                            "pre": coef, "post": coef, "residual": coef, "ids": exact,
                            "ws": ratio_allclose(atol=2e-4, rtol=2e-3, max_error_ratio=0, valid_rows=active)},
                compile_only=args.compile_only,
                runtime_dir=artifact,
                a5_default_runtime="host_build_graph",
                config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device, enable_chip_swimlane=1),
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
            if args.compile_only:
                break
            artifact = result.work_dir
            check_dispatched_path(artifact, before, rows=rows, active=active, cluster_limit=cluster_limit)


    main()
