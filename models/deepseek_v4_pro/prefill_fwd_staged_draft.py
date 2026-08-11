# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""Memory-bounded DeepSeek-V4 Pro packed-prefill forward bring-up.

This driver executes all 61 ``prefill_layer`` stages in model order while only
one layer's synthetic weights are live. It chains ``x_next`` between layers and
keeps each layer's mutable attention state for subsequent rounds. The default
fixture is one 128-token request, which is the smallest first-bring-up case.

After layer 60, the main-model Hyper-Connections head and final RMSNorm run in
fixed-size packed-token tiles. There is intentionally no golden comparison,
embedding, MTP, LM head, or logits stage. Expert payloads use the current
shape-correct INT8 smoke stand-in.
"""

import argparse
import gc
import math

import torch
from config import PRO_KERNEL as MODEL_CONFIG
from golden import run_jit
from prefill_layer import N_RANKS, T, build_tensor_specs, l3_prefill_layer
from pypto.ir.distributed_compiled_program import DistributedConfig
from staged_fwd_tail import (
    build_tensor_specs as build_fwd_tail_tensor_specs,
    l3_prefill_fwd_tail,
)
from staged_fwd_utils import override_inputs


RNG_CHECKPOINT_NAMES = frozenset({
    "kv_cache",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state",
    "csa_compress_state",
    "csa_inner_compress_state",
})


def _specialization_key(layer_id):
    """Return the attention/routing branch pair constant-folded by JIT."""
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    attention = {128: "hca", 4: "csa"}[ratio]
    routing = "hash" if layer_id < MODEL_CONFIG.num_hash_layers else "score"
    return attention, routing


def _specialization_representatives():
    representatives = {}
    for layer_id in range(MODEL_CONFIG.num_hidden_layers):
        representatives.setdefault(_specialization_key(layer_id), layer_id)
    expected = {("hca", "hash"), ("csa", "hash"), ("hca", "score"), ("csa", "score")}
    if set(representatives) != expected:
        raise ValueError(f"unexpected prefill specialization classes: {representatives}")
    return representatives


def _active_cache_names(layer_id):
    if MODEL_CONFIG.compress_ratios[layer_id] == 128:
        return frozenset({"kv_cache", "cmp_kv", "hca_compress_state"})
    return frozenset({
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
        "csa_compress_state",
        "csa_inner_compress_state",
    })


def _round_start_positions(base_positions, chunk_lens, round_id):
    return tuple(
        start_pos + round_id * chunk_len
        for start_pos, chunk_len in zip(base_positions, chunk_lens)
    )


def _run_compile_only(args, device_ids, representatives, chunk_lens, start_positions):
    runtime_cfg = {
        "platform": args.platform,
        "enable_l2_swimlane": args.enable_l2_swimlane,
        "startup_timeout_s": args.startup_timeout_s,
    }
    compile_cfg = {
        "dump_passes": args.dump_passes,
        "distributed_config": DistributedConfig(
            device_ids=device_ids[:N_RANKS],
            num_sub_workers=0,
        ),
    }
    for key, layer_id in representatives.items():
        print(f"[STAGED] compiling {key[0]}+{key[1]} with representative layer {layer_id}", flush=True)
        torch.manual_seed(args.seed + layer_id)
        specs = build_tensor_specs(
            layer_id=layer_id,
            chunk_lens=chunk_lens,
            start_positions=start_positions,
            smoke_weights=True,
        )
        result = run_jit(
            fn=l3_prefill_layer,
            specs=specs,
            golden_fn=None,
            compile_only=True,
            save_data=False,
            compile_cfg=compile_cfg,
            runtime_cfg=runtime_cfg,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            return False
        del result, specs
        gc.collect()

    print("[STAGED] compiling hc_head+final_rmsnorm tail", flush=True)
    torch.manual_seed(args.seed + MODEL_CONFIG.num_hidden_layers)
    specs = build_fwd_tail_tensor_specs(T)
    result = run_jit(
        fn=l3_prefill_fwd_tail,
        specs=specs,
        golden_fn=None,
        compile_only=True,
        save_data=False,
        compile_cfg=compile_cfg,
        runtime_cfg=runtime_cfg,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        return False
    del result, specs
    gc.collect()
    return True


def _run_staged(args, device_ids, representatives, chunk_lens, base_positions):
    runtime_dirs = {}
    tail_runtime_dir = None
    caches_by_layer = {}
    cache_rng_states_by_layer = {}
    compile_cfg = {
        "dump_passes": args.dump_passes,
        "distributed_config": DistributedConfig(
            device_ids=device_ids[:N_RANKS],
            num_sub_workers=0,
        ),
    }
    runtime_cfg = {
        "platform": args.platform,
        "enable_l2_swimlane": args.enable_l2_swimlane,
        "startup_timeout_s": args.startup_timeout_s,
    }

    for round_id in range(args.rounds):
        hidden = None
        start_positions = _round_start_positions(base_positions, chunk_lens, round_id)
        print(
            f"[STAGED] prefill round {round_id + 1}/{args.rounds}, "
            f"start_positions={','.join(str(pos) for pos in start_positions)}",
            flush=True,
        )
        for layer_id in range(MODEL_CONFIG.num_hidden_layers):
            key = _specialization_key(layer_id)
            representative = representatives[key]
            runtime_dir = runtime_dirs.get(key)
            action = "compile" if runtime_dir is None else "reuse"
            print(
                f"[STAGED] layer {layer_id:02d}/{MODEL_CONFIG.num_hidden_layers - 1}: "
                f"{key[0]}+{key[1]} ({action} L{representative})",
                flush=True,
            )

            torch.manual_seed(args.seed + layer_id)
            specs = build_tensor_specs(
                layer_id=layer_id,
                chunk_lens=chunk_lens,
                start_positions=start_positions,
                smoke_weights=True,
            )
            overrides = dict(caches_by_layer.get(layer_id, {}))
            if hidden is not None:
                overrides["x_hc"] = hidden
            cache_rng_states = cache_rng_states_by_layer.setdefault(layer_id, {})
            override_inputs(
                specs,
                overrides,
                tracked_names=RNG_CHECKPOINT_NAMES,
                rng_after_init=cache_rng_states,
            )

            result = run_jit(
                fn=l3_prefill_layer,
                specs=specs,
                golden_fn=None,
                compile_only=False,
                runtime_dir=None if runtime_dir is None else str(runtime_dir),
                save_data=False,
                return_outputs=True,
                compile_cfg=compile_cfg,
                runtime_cfg=runtime_cfg,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                return False
            if result.work_dir is None or result.outputs is None:
                raise RuntimeError(f"layer {layer_id} completed without a runtime directory or outputs")
            runtime_dirs.setdefault(key, result.work_dir)

            outputs = result.outputs
            required = {"x_next", *_active_cache_names(layer_id)}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(f"layer {layer_id} did not return required outputs: {sorted(missing)}")
            hidden = outputs["x_next"]
            caches_by_layer[layer_id] = {
                name: outputs[name]
                for name in _active_cache_names(layer_id)
            }

            del outputs, result, overrides, specs
            gc.collect()

        if hidden is None:
            raise RuntimeError("prefill staged layers completed without a hidden state")
        if hidden.shape[1] % T != 0:
            raise RuntimeError(
                f"prefill tail requires whole {T}-token physical tiles, got {hidden.shape[1]} tokens"
            )
        normalized_tokens = 0
        for tile_start in range(0, hidden.shape[1], T):
            action = "compile" if tail_runtime_dir is None else "reuse"
            print(
                f"[STAGED] hc_head+final_rmsnorm tail tile {tile_start // T} ({action})",
                flush=True,
            )
            torch.manual_seed(args.seed + MODEL_CONFIG.num_hidden_layers)
            specs = build_fwd_tail_tensor_specs(T)
            override_inputs(
                specs,
                {"x_hc": hidden[:, tile_start : tile_start + T].contiguous()},
            )
            result = run_jit(
                fn=l3_prefill_fwd_tail,
                specs=specs,
                golden_fn=None,
                compile_only=False,
                runtime_dir=None if tail_runtime_dir is None else str(tail_runtime_dir),
                save_data=False,
                return_outputs=True,
                compile_cfg=compile_cfg,
                runtime_cfg=runtime_cfg,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                return False
            if result.work_dir is None or result.outputs is None:
                raise RuntimeError("prefill tail completed without a runtime directory or outputs")
            tail_runtime_dir = result.work_dir
            if "hidden_out" not in result.outputs:
                raise RuntimeError("prefill tail did not return hidden_out")
            normalized_tokens += result.outputs["hidden_out"].shape[1]
            del result, specs
            gc.collect()

        del hidden
        gc.collect()
        print(
            f"[STAGED] prefill round {round_id + 1}/{args.rounds} complete, "
            f"normalized_physical_tokens={normalized_tokens}",
            flush=True,
        )
    return True


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Pro staged 61-layer packed-prefill driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--chunk-lens", type=str, default=str(T),
                        help="comma-separated per-request logical chunk lengths")
    parser.add_argument("--start-positions", type=str, default=None,
                        help="comma-separated prior context lengths; defaults to all zeros")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed used to regenerate stable per-layer smoke weights")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1,
                        default=0, choices=(0, 1, 2))
    parser.add_argument("--startup-timeout-s", type=float, default=900.0,
                        help="forked EP worker startup deadline in seconds")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if not math.isfinite(args.startup_timeout_s) or args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive and finite")
    if args.ep != N_RANKS:
        parser.error(f"--ep was imported as {N_RANKS}, got parsed value {args.ep}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        parser.error(f"need at least {N_RANKS} devices, got {device_ids}")

    chunk_lens = tuple(int(value) for value in args.chunk_lens.split(","))
    if not chunk_lens or any(chunk_len <= 0 for chunk_len in chunk_lens):
        parser.error(f"--chunk-lens must contain positive integers, got {chunk_lens}")
    if args.start_positions is None:
        base_positions = (0,) * len(chunk_lens)
    else:
        base_positions = tuple(int(value) for value in args.start_positions.split(","))
    if len(base_positions) != len(chunk_lens):
        parser.error("--start-positions must have the same number of values as --chunk-lens")
    if any(start_pos < 0 for start_pos in base_positions):
        parser.error(f"--start-positions must be non-negative, got {base_positions}")

    representatives = _specialization_representatives()
    passed = (
        _run_compile_only(
            args,
            device_ids,
            representatives,
            chunk_lens,
            base_positions,
        )
        if args.compile_only
        else _run_staged(
            args,
            device_ids,
            representatives,
            chunk_lens,
            base_positions,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
