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
"""Memory-bounded DeepSeek-V4 Pro decode forward bring-up.

The monolithic forward stacks all 61 layers' synthetic weights in one host
fixture. This driver instead runs ``decode_layer`` once per layer, immediately
releases that layer's weights, and chains ``x_next`` into the following layer.
Mutable attention caches are retained per layer and restored in later rounds.

After layer 60, the main-model Hyper-Connections head and final RMSNorm run as
a separate staged program. There is intentionally no golden comparison,
embedding, MTP, LM head, or logits stage. Expert payloads use the current
shape-correct INT8 smoke stand-in.
"""

import argparse
import gc
import math

import torch
from config import DECODE_SEQ, DECODE_START_POS, DECODE_TOKENS, PRO_KERNEL as MODEL_CONFIG
from decode_layer import N_RANKS, build_tensor_specs, l3_decode_layer
from golden import run_jit
from pypto.ir.distributed_compiled_program import DistributedConfig
from staged_fwd_tail import (
    build_tensor_specs as build_fwd_tail_tensor_specs,
    l3_decode_fwd_tail,
)
from staged_fwd_utils import override_inputs


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
        raise ValueError(f"unexpected decode specialization classes: {representatives}")
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


def _run_compile_only(args, device_ids, representatives):
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
            start_pos=args.start_pos,
            layer_id=layer_id,
            smoke_weights=True,
            cache_outputs=True,
        )
        result = run_jit(
            fn=l3_decode_layer,
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
    specs = build_fwd_tail_tensor_specs(DECODE_TOKENS)
    result = run_jit(
        fn=l3_decode_fwd_tail,
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


def _run_staged(args, device_ids, representatives):
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
        start_pos = args.start_pos + round_id * DECODE_SEQ
        print(
            f"[STAGED] decode round {round_id + 1}/{args.rounds}, start_pos={start_pos}",
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
                start_pos=start_pos,
                layer_id=layer_id,
                smoke_weights=True,
                cache_outputs=True,
            )
            overrides = dict(caches_by_layer.get(layer_id, {}))
            if hidden is not None:
                overrides["x_hc"] = hidden
            cache_rng_states = cache_rng_states_by_layer.setdefault(layer_id, {})
            override_inputs(
                specs,
                overrides,
                tracked_names=_active_cache_names(layer_id),
                rng_after_init=cache_rng_states,
            )

            result = run_jit(
                fn=l3_decode_layer,
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

            # Keep only the chained hidden state and active caches. All synthetic
            # layer weights become unreachable before the next layer is built.
            del outputs, result, overrides, specs
            gc.collect()

        if hidden is None:
            raise RuntimeError("decode staged layers completed without a hidden state")
        action = "compile" if tail_runtime_dir is None else "reuse"
        print(f"[STAGED] hc_head+final_rmsnorm tail ({action})", flush=True)
        torch.manual_seed(args.seed + MODEL_CONFIG.num_hidden_layers)
        specs = build_fwd_tail_tensor_specs(DECODE_TOKENS)
        override_inputs(specs, {"x_hc": hidden})
        result = run_jit(
            fn=l3_decode_fwd_tail,
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
            raise RuntimeError("decode tail completed without a runtime directory or outputs")
        tail_runtime_dir = result.work_dir
        if "hidden_out" not in result.outputs:
            raise RuntimeError("decode tail did not return hidden_out")
        normalized_shape = tuple(result.outputs["hidden_out"].shape)
        del result, specs, hidden
        gc.collect()

        print(
            f"[STAGED] decode round {round_id + 1}/{args.rounds} complete, "
            f"normalized_hidden_shape={normalized_shape}",
            flush=True,
        )
    return True


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Pro staged 61-layer decode driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=DECODE_START_POS)
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

    representatives = _specialization_representatives()
    passed = (
        _run_compile_only(args, device_ids, representatives)
        if args.compile_only
        else _run_staged(args, device_ids, representatives)
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
