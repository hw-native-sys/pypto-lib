# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the real compressed-tensors loader and both TND weight adapters."""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["spmd", "mpmd", "both"], default="both")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    report = {"passed": False, "scope": "real layer 4 attention load and binding; no model generation"}
    try:
        os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
        os.environ["PTO_CSA_VARIANT"] = "mpmd_tnd"
        os.environ["PTO_CSA_RUNTIME"] = "tensormap_and_ringbuffer"
        import torch
        import torch_npu
        from safetensors import safe_open
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from models.vllm_ascend.deepseek_v4_flash.csa.serving_fixture import DeepseekCompressedOverrides

        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        current_platform.pre_register_and_update()
        config = EngineArgs(
            model=str(args.model),
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            quantization="compressed-tensors",
            dtype="bfloat16",
            hf_overrides=DeepseekCompressedOverrides(),
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=args.history + 256,
            max_num_seqs=args.batch,
            max_num_batched_tokens=256,
            block_size=32,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={
                "weight_nz_mode": 2,
                "enable_kv_nz": False,
                "enable_dsa_cp": False,
                "enable_cpu_binding": False,
            },
        ).create_engine_config()
        from models.vllm_ascend.deepseek_v4_flash.csa.serving_fixture import native_session
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM, DeepseekV4Attention
        from vllm_ascend.utils import enable_custom_op, register_ascend_customop

        if not enable_custom_op():
            raise RuntimeError("The vLLM Ascend custom operator library is required for Native metadata")

        device = torch.device(f"npu:{args.device}")
        index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        names = sorted(
            name
            for name in index
            if name.startswith("layers.4.attn.")
            or name
            in (
                "layers.4.attn_norm.weight",
                "layers.4.hc_attn_fn",
                "layers.4.hc_attn_base",
                "layers.4.hc_attn_scale",
            )
        )

        def weights():
            for name in names:
                with safe_open(args.model / index[name], framework="pt", device="cpu") as reader:
                    yield name, reader.get_tensor(name)

        with native_session(config, args.device), torch.inference_mode():
            register_ascend_customop(config)
            # Keep the real model's parameter names and load_weights implementation,
            # but instantiate only the attention half of layer 4 (no MoE allocation).
            holder = torch.nn.Module()
            holder.config = config.model_config.hf_config
            holder.quant_config = config.quant_config
            holder.num_redundant_experts = 0
            holder.model = torch.nn.Module()
            layer = torch.nn.Module()
            holder.model.layers = torch.nn.ModuleDict({"4": layer})
            prior_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device(device):
                    layer.self_attn = DeepseekV4Attention(
                        config,
                        holder.config,
                        max_position_embeddings=holder.config.rope_parameters[
                            "original_max_position_embeddings"
                        ],
                        cache_config=config.cache_config,
                        quant_config=config.quant_config,
                        prefix="model.layers.4.self_attn",
                    )
                    layer.input_layernorm = RMSNorm(holder.config.hidden_size, eps=holder.config.rms_norm_eps)
                for key in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base"):
                    name = "layers.4." + key
                    with safe_open(args.model / index[name], framework="pt", device="cpu") as reader:
                        value = reader.get_tensor(name)
                    setattr(layer, key, torch.nn.Parameter(torch.empty_like(value, device=device)))
            finally:
                torch.set_default_dtype(prior_dtype)
            loaded = AscendDeepseekV4ForCausalLM.load_weights(holder, weights())
            report["source_tensors"] = names
            report["loaded_parameters"] = sorted(loaded)
            missing = set(dict(holder.named_parameters())) - loaded
            report["computed_parameters"] = sorted(missing)
            if any(not name.endswith(".weight_offset") for name in missing):
                raise ValueError(f"Unexpected unloaded parameters: {sorted(missing)}")
            for name, value in holder.named_parameters():
                if value.is_floating_point() and not torch.isfinite(value).all().item():
                    raise ValueError(f"Nonfinite loaded parameter: {name}")
                if name in missing and torch.count_nonzero(value).item():
                    raise ValueError(f"Nonzero symmetric offset: {name}")
            process_weights_after_loading(holder, config.model_config, device)
            from importlib import import_module
            from vllm_ascend.ops.pypto.variant import variant_package

            os.environ["PTO_CSA_VARIANT"] = "mpmd_tnd"
            assert variant_package().startswith("models.vllm_ascend.deepseek_v4_flash.csa.")
            MPMD = import_module(variant_package() + ".service").CSAServiceRuntime
            os.environ["PTO_CSA_VARIANT"] = "spmd_tnd"
            assert variant_package().startswith("models.vllm_ascend.deepseek_v4_flash.csa.")
            SPMD = import_module(variant_package() + ".service").CSAServiceRuntime
            os.environ["PTO_CSA_VARIANT"] = "spmd_tnd" if args.mode == "spmd" else "mpmd_tnd"

            prepared = {
                "mpmd": MPMD._prepare_weights(layer.self_attn, layer),
                "spmd": SPMD._prepare_weights(layer.self_attn, layer),
            }
            if prepared["mpmd"].keys() != prepared["spmd"].keys():
                raise ValueError("Different weight adapter key sets")
            report["weights"] = {}
            for name, value in prepared["mpmd"].items():
                other = prepared["spmd"][name]
                equal = torch.equal(
                    torch_npu.npu_format_cast(value, 2).cpu(), torch_npu.npu_format_cast(other, 2).cpu()
                )
                report["weights"][name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "format": int(torch_npu.get_npu_format(value)),
                    "spmd_format": int(torch_npu.get_npu_format(other)),
                    "equal": equal,
                }
                if not equal:
                    raise ValueError(f"Different adapted weights: {name}")
                if name in ("wq_a", "wq_b", "wo_a", "wo_b") and any(
                    int(torch_npu.get_npu_format(tensor)) != 29 for tensor in (value, other)
                ):
                    raise ValueError(f"TND root weight lacks NZ format: {name}")
            report["weights_loading_passed"] = True
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            report["forward"] = run_forward(config, layer, args)
            report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


def run_forward(config, layer, args):
    from importlib import import_module
    from types import MethodType

    import torch
    import pypto.torch
    from pypto.runtime.kernel.context import get_process_kernel_state
    from golden import validate_golden
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.models.deepseek_v4 import DeepseekV2DecoderLayer
    from vllm_ascend.ops.dsa import _build_kv_cache
    from vllm_ascend.ops.pypto.variant import variant_package, ring_sizing_kwargs
    from models.vllm_ascend.deepseek_v4_flash.csa.serving_fixture import make_fixture, collect_state, restore

    hf = config.model_config.hf_config
    layer.hc_mult, layer.hc_sinkhorn_iters = hf.hc_mult, hf.hc_sinkhorn_iters
    layer.norm_eps, layer.hc_eps = hf.rms_norm_eps, hf.hc_eps
    layer.hc_pre = MethodType(DeepseekV2DecoderLayer.hc_pre, layer)
    layer.hc_post = MethodType(DeepseekV2DecoderLayer.hc_post, layer)
    wrapper = layer.self_attn.dsa_attn
    wrapper._pto_csa_layer = (layer,)
    fixture = make_fixture(
        config, layer.self_attn, args.batch, args.history, args.seed, torch.device(f"npu:{args.device}")
    )
    output = torch.empty_like(fixture["hidden"])
    pypto.torch.init(
        device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer", **ring_sizing_kwargs()
    )
    modes = ("spmd", "mpmd") if args.mode == "both" else (args.mode,)
    first = None
    report = {}

    def bits(actual, expected):
        return {name: value.contiguous().view(torch.uint8) for name, value in actual.items()}, {
            name: value.contiguous().view(torch.uint8) for name, value in expected.items()
        }

    for mode in modes:
        os.environ["PTO_CSA_VARIANT"] = mode + "_tnd"
        package = variant_package()
        if not package.startswith("models.vllm_ascend.deepseek_v4_flash.csa."):
            raise RuntimeError("Configure the vLLM Ascend CSA variant selector to use pypto-lib first")
        runtime = import_module(package + ".service").CSAServiceRuntime(
            layer.self_attn, None, args.batch, layer
        )
        previous = None
        for repeat in range(2):
            restore(fixture)
            with set_ascend_forward_context(
                fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
            ):
                context = get_forward_context()
                if not runtime.eligible(context, fixture["hidden"], fixture["positions"]):
                    raise RuntimeError("Service rejected the valid decode descriptor")
                runtime(
                    context,
                    fixture["hidden"],
                    fixture["positions"],
                    output,
                    _build_kv_cache(wrapper, context),
                )
            torch.npu.synchronize()
            # Retire native RunHandles, not only Torch stream work, before the
            # next mode registers a new callable on the pinned eager bridge.
            get_process_kernel_state().drain()
            topk, scores = runtime._outputs[(fixture["tokens"], output.device)]
            state = collect_state(fixture, output, topk)
            state["idx_topk_scores"] = scores.detach().cpu()
            for name, group in fixture["groups"].items():
                changed = group["allocation"].cpu() != group["initial"]
                if (changed & ~group["allowed"]).any():
                    raise AssertionError(f"Unexpected cache write: {name}")
            for name, (value, initial) in fixture.get("readonly", {}).items():
                validate_golden({name: value.cpu()}, {name: initial}, rtol=0, atol=0)
            if previous is not None:
                actual_bits, expected_bits = bits(state, previous)
                validate_golden(actual_bits, expected_bits, rtol=0, atol=0)
            previous = state
        if first is not None:
            actual_bits, expected_bits = bits(state, first)
            validate_golden(actual_bits, expected_bits, rtol=0, atol=0)
        first = state
        torch.save(state, args.output / (mode + "_attention_state.pt"))
        report[mode] = {"passed": True, "package": package, "repeat_bit_equal": True, "cache_guards": True}
    return {
        "scope": "real-weight serving adapter and state equivalence; not full-model generation",
        "modes": report,
        "mpmd_spmd_bit_equal": True if args.mode == "both" else "NOT_RUN",
    }


if __name__ == "__main__":
    raise SystemExit(main())
