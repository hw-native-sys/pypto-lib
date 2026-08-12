# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent two-card regression for the DeepSeek-V4 MoE window protocol.

The ordinary ``moe.py`` test performs one MoE call and validates one launch.
That does not exercise stale-window or cross-epoch ordering bugs. This test:

* issues ten dependent MoE epochs inside one rank-generic device graph;
* reuses one set of communication windows and clears it only after the graph;
* alternates sharply different self/remote expert layouts between host rounds;
* reuses one persistent prepared worker with window reset disabled; and
* validates the final three window states against a recursively chained torch golden.

Run on two cards, for example::

    python models/deepseek_v4_flash_mtp/moe_protocol_stress.py \
        -p a2a3 --ep 2 -d 0,1 --rounds 20
"""

import argparse
import dataclasses
import sys
import time

import torch

import config


def _parse_ep_argv():
    for i, tok in enumerate(sys.argv):
        if tok == "--ep" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--ep="):
            return int(tok.split("=", 1)[1])
    return 2


# ``moe`` and its leaf kernels freeze these values while importing.
EP = _parse_ep_argv()
config.EP_WORLD_SIZE = EP
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=config.FLASH.n_routed_experts // 8 * EP)
if "--prefill" in sys.argv:
    config.MOE_TOKENS = config.PREFILL_TOKENS
config.RECV_MAX = EP * config.MOE_TOKENS

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

from golden import ScalarSpec, TensorSpec, ratio_reldiff, validate_golden
from golden.runner import _l3_ordered_names, _l3_run_config, _share_in_place
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs,
    clear_moe_signals,
    golden_moe,
    moe,
)


STRESS_STEPS = 3
STRESS_TOKENS = (T, T, T)
STRESS_PAIRS = 4
STRESS_EPOCHS = 2 * STRESS_PAIRS + 2
OUTPUT_EPOCHS = (STRESS_EPOCHS - 1, STRESS_EPOCHS - 2, STRESS_EPOCHS)

assert N_RANKS == 2, "the protocol stress regression intentionally targets EP=2"
assert T >= 2
assert TOPK % 2 == 0
assert N_LOCAL >= TOPK


@pl.jit(auto_scope=False)
def moe_protocol_stress_rank(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids_0: pl.Tensor[[T], pl.INT64],
    input_ids_1: pl.Tensor[[T], pl.INT64],
    input_ids_2: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next_0: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_next_1: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_next_2: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 2], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 2], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Issue ten dependent MoE epochs inside one device graph."""
    num_tokens = pl.const(T, pl.INT32)
    moe(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids_0,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale, x_next_0,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived,
        layer_id, num_tokens, pl.const(1, pl.INT32), pl.const(1, pl.INT32),
        my_rank, pl.const(1, pl.INT32),
    )
    for pair in pl.range(STRESS_PAIRS):
        even_epoch = pl.cast(pair * 2 + 2, pl.INT32)
        odd_epoch = pl.cast(pair * 2 + 3, pl.INT32)
        moe(
            x_next_0, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids_1,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, x_next_1,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, num_tokens, pl.const(1, pl.INT32), pl.const(1, pl.INT32),
            my_rank, even_epoch,
        )
        moe(
            x_next_1, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids_2,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, x_next_0,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, num_tokens, pl.const(1, pl.INT32), pl.const(1, pl.INT32),
            my_rank, odd_epoch,
        )
    moe(
        x_next_0, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids_0,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale, x_next_2,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived,
        layer_id, num_tokens, pl.const(1, pl.INT32), pl.const(1, pl.INT32),
        my_rank, pl.const(STRESS_EPOCHS, pl.INT32),
    )
    clear_moe_signals(x_next_2, recv_meta, arrived, data_arrived, combine_arrived)


@pl.jit.host
def l3_moe_protocol_stress(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, STRESS_STEPS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[STRESS_STEPS, N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    """Issue ten epochs against the same distributed windows before clearing."""
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 2], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 2], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 2], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 2], dtype=pl.INT32)
        moe_protocol_stress_rank(
            x_hc[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank, 0], input_ids[rank, 1], input_ids[rank, 2],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_next[0, rank], x_next[1, rank], x_next[2, rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, rank,
            device=rank,
        )


def _route_row(rank, pattern, token):
    """Return distinct routes split evenly between the local and remote rank."""
    half = TOPK // 2
    remote = 1 - rank
    if pattern == 0:
        local_base = token % half
        remote_base = (token * 3) % half
    else:
        # Disjoint local-expert bands make the packed per-expert count words very
        # different from pattern 0 while retaining both self and remote traffic.
        local_base = N_LOCAL // 2 + token % half
        remote_base = N_LOCAL - half - (token % half)
    local = [rank * N_LOCAL + (local_base + k) % N_LOCAL for k in range(half)]
    remote_routes = [remote * N_LOCAL + (remote_base + k) % N_LOCAL for k in range(half)]
    return local + remote_routes


def _routing_fixtures():
    """Build A/B/A and B/A/B route sequences for alternating host rounds."""
    tid2eid = torch.zeros(N_RANKS, VOCAB, TOPK, dtype=torch.int32)
    variants = []
    next_row = 0
    for sequence in ((0, 1, 0), (1, 0, 1)):
        input_ids = torch.zeros(N_RANKS, STRESS_STEPS, T, dtype=torch.int64)
        for step, pattern in enumerate(sequence):
            for rank in range(N_RANKS):
                for token in range(T):
                    row = next_row
                    next_row += 1
                    tid2eid[rank, row] = torch.tensor(_route_row(rank, pattern, token), dtype=torch.int32)
                    input_ids[rank, step, token] = row
        variants.append(input_ids)
    return tid2eid, variants


def _build_stress_specs():
    tid2eid, input_variants = _routing_fixtures()
    specs = build_tensor_specs(layer_id=0, num_tokens=T)
    for spec in specs:
        if spec.name == "tid2eid":
            spec.init_value = tid2eid
        elif spec.name == "input_ids":
            spec.shape = [N_RANKS, STRESS_STEPS, T]
            spec.init_value = input_variants[0]
        elif spec.name == "x_next":
            spec.shape = [STRESS_STEPS, N_RANKS, T, HC_MULT, D]
    return [spec for spec in specs if spec.name != "num_tokens"], input_variants


def _golden_for(tensors, input_ids, x_hc):
    expected = torch.zeros(STRESS_STEPS, N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    epoch_x_hc = x_hc
    output_slot = {epoch: slot for slot, epoch in enumerate(OUTPUT_EPOCHS)}
    for epoch in range(1, STRESS_EPOCHS + 1):
        route_step = 0 if epoch in (1, STRESS_EPOCHS) else (1 if epoch % 2 == 0 else 2)
        epoch_output = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        values = {name: value for name, value in tensors.items() if name != "x_next"}
        values["x_hc"] = epoch_x_hc
        values["input_ids"] = input_ids[:, route_step]
        values["layer_id"] = 0
        values["num_tokens"] = T
        values["x_next"] = epoch_output
        golden_moe(values)
        if epoch in output_slot:
            expected[output_slot[epoch]] = epoch_output
        epoch_x_hc = epoch_output
    return expected


def _compile(specs, platform, device_ids):
    dummy_args = [
        spec.value.item() if isinstance(spec, ScalarSpec)
        else torch.empty(spec.shape, dtype=spec.dtype)
        for spec in specs
    ]
    return l3_moe_protocol_stress.compile(
        *dummy_args,
        config=RunConfig(
            platform=platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        ),
    )


def run_stress(
    platform,
    device_ids,
    rounds,
    seed,
    reset_windows=False,
    fixed_variant=None,
    fixed_routes=None,
    fixed_activation=None,
    log_level=None,
):
    torch.manual_seed(seed)
    specs, input_variants = _build_stress_specs()
    tensor_specs = [spec for spec in specs if isinstance(spec, TensorSpec)]
    scalar_specs = {spec.name: spec for spec in specs if isinstance(spec, ScalarSpec)}
    tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
    base_x_hc = tensors["x_hc"].clone()
    activation_variants = [base_x_hc, -base_x_hc]

    print("[STRESS] compile ...", flush=True)
    compiled = _compile(specs, platform, device_ids)
    print(f"[STRESS] compile done: {compiled.output_dir}", flush=True)

    print("[STRESS] compute route/activation golden matrix ...", flush=True)
    goldens = {
        (route_variant, activation_variant): _golden_for(tensors, ids, x_hc)
        for route_variant, ids in enumerate(input_variants)
        for activation_variant, x_hc in enumerate(activation_variants)
    }
    _share_in_place(tensors)
    shared_input_variants = [ids.contiguous().share_memory_() for ids in input_variants]
    shared_activation_variants = [x.contiguous().share_memory_() for x in activation_variants]

    ordered_names = _l3_ordered_names(compiled)
    runtime_cfg = {"platform": platform}
    run_config = _l3_run_config(runtime_cfg)
    if log_level is not None:
        from pypto.runtime.log_config import configure_log
        configure_log(log_level)

    resident_specs = [spec for spec in tensor_specs if spec.is_resident]
    resident_handles = []
    started = time.time()
    with compiled.prepare(
        run_config,
        persistent=True,
        reset_persistent_windows=reset_windows,
    ) as worker:
        try:
            resident_args = {}
            for spec in resident_specs:
                if spec.resident != "stacked":
                    raise ValueError(f"stress runner only supports stacked resident tensors: {spec.name}")
                handle = worker.alloc_stacked_tensor(tensors[spec.name])
                resident_handles.append((spec.name, handle))
                resident_args[spec.name] = handle

            call_overrides = {}

            def arg(name):
                if name in call_overrides:
                    return call_overrides[name]
                if name in resident_args:
                    return resident_args[name]
                if name in tensors:
                    return tensors[name]
                return scalar_specs[name].value

            for round_id in range(rounds):
                variant = fixed_variant if fixed_variant is not None else round_id % len(input_variants)
                route_variant = fixed_routes if fixed_routes is not None else variant
                activation_variant = fixed_activation if fixed_activation is not None else variant
                # Bind the selected shared input buffer for this dispatch. Simpler
                # prebuilds small tensor bindings, so mutating one already-bound
                # INT64 buffer in place can intentionally reuse its old snapshot;
                # serving uses stable double-buffered input objects in the same way.
                call_overrides["input_ids"] = shared_input_variants[route_variant]
                tensors["input_ids"] = call_overrides["input_ids"]
                # Alternating a zero-mean fixture and its negation prevents a
                # stale output from matching the next launch without changing
                # the calibrated input distribution of the ordinary MoE test.
                tensors["x_hc"].copy_(shared_activation_variants[activation_variant])
                ordered = [arg(name) for name in ordered_names]
                worker(*ordered, config=run_config)

                for step in range(STRESS_STEPS):
                    output_name = f"x_next_epoch_{OUTPUT_EPOCHS[step]}"
                    active_tokens = STRESS_TOKENS[step]
                    actual = tensors["x_next"][step, :, :active_tokens]
                    expected_full = goldens[(route_variant, activation_variant)][step]
                    expected = expected_full[:, :active_tokens]
                    try:
                        validate_golden(
                            {output_name: actual},
                            {output_name: expected},
                            rtol=1e-3,
                            atol=1e-3,
                            compare_fn={output_name: ratio_reldiff(diff_thd=3e-3, pct_thd=0.05)},
                            inputs={
                                "input_ids": tensors["input_ids"][:, step, :active_tokens],
                                "x_hc": tensors["x_hc"][:, :active_tokens],
                            },
                        )
                    except AssertionError:
                        print(
                            f"[STRESS] round {round_id + 1} "
                            f"epoch {OUTPUT_EPOCHS[step]} mismatch diagnostics:"
                        )
                        for candidate, candidate_golden in goldens.items():
                            delta = (actual - candidate_golden[step, :, :active_tokens]).abs()
                            print(
                                f"[STRESS]   golden_route_activation={candidate} "
                                f"mean_abs={delta.mean().item():.6g} max_abs={delta.max().item():.6g}",
                                flush=True,
                            )
                        raise
                print(
                    f"[STRESS] round {round_id + 1}/{rounds} "
                    f"route={route_variant} activation={activation_variant} "
                    "PASS",
                    flush=True,
                )
        finally:
            for _, handle in reversed(resident_handles):
                worker.free_stacked_tensor(handle)

    print(f"[STRESS] PASS: {rounds} persistent rounds in {time.time() - started:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=2, choices=[2])
    parser.add_argument("-d", "--device", default="0,1")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="exercise the 128-token prefill MoE shape instead of decode shape",
    )
    parser.add_argument(
        "--reset-windows",
        action="store_true",
        help="runtime-reset communication windows between launches (diagnostic control)",
    )
    parser.add_argument(
        "--fixed-variant",
        type=int,
        choices=[0, 1],
        default=None,
        help="reuse one input fixture instead of alternating A/B (diagnostic control)",
    )
    parser.add_argument(
        "--fixed-routes",
        type=int,
        choices=[0, 1],
        default=None,
        help="hold input_ids/routes fixed while activation inputs continue alternating",
    )
    parser.add_argument(
        "--fixed-activation",
        type=int,
        choices=[0, 1],
        default=None,
        help="hold x_hc fixed while input_ids/routes continue alternating",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        parser.error(f"need exactly {N_RANKS} devices, got {device_ids}")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    run_stress(
        args.platform,
        device_ids,
        args.rounds,
        args.seed,
        reset_windows=args.reset_windows,
        fixed_variant=args.fixed_variant,
        fixed_routes=args.fixed_routes,
        fixed_activation=args.fixed_activation,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
