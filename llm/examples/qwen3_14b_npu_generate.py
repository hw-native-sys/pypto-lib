# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# R4 mitigation: PTO2_RING_HEAP nominal value is multiplied ×4 by the
# runtime, so 2 GiB nominal → 8 GiB actual GM heap. This leaves HBM
# headroom for the all-layers stacked weights (~2.14 GiB) + KV cache
# (~4 GiB) + shared mem (~23 GiB) on a 62 GiB device. Override via env
# in the launching shell if you need a different value.
os.environ.setdefault("PTO2_RING_HEAP", str(2 * 1024 ** 3))


def _bootstrap_package_root() -> None:
    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parents[1],
        this_file.parents[1] / "llm",
    ]
    for package_dir in candidates:
        if (package_dir / "__init__.py").exists() and (package_dir / "core").is_dir():
            package_parent = package_dir.parent
            package_parent_str = str(package_parent)
            if package_parent_str not in sys.path:
                sys.path.insert(0, package_parent_str)
            return
    raise RuntimeError(f"Unable to locate the llm package root from {this_file}")


_bootstrap_package_root()

from llm.core import GenerateConfig, LLMEngine, RuntimeConfig
from llm.core.kv_cache import KvCacheManager
from llm.core.pypto_executor import PyptoQwen14BExecutor
from llm.core.types import LoadedModel
import dataclasses


# -----------------------------------------------------------------------------
# Profiling helpers (3-level: phase / executor API / per-layer kernel)
# -----------------------------------------------------------------------------


class _TimingCollector:
    """Collects timings at three levels and groups per-layer kernel calls per
    decode step so the report can show per-step breakdowns.
    """

    def __init__(self) -> None:
        # name -> total seconds (additive across calls)
        self.phases: dict[str, float] = {}
        # kernel_name -> flat list of seconds (one entry per kernel invocation)
        self.kernel_times: dict[str, list[float]] = defaultdict(list)
        # kernel_name -> list[list[float]]: outer = decode step, inner = layer
        self.kernel_per_decode_step: dict[str, list[list[float]]] = defaultdict(list)
        # Bumped by BeginDecodeStep before each run_decode invocation.
        self._decode_step_idx: int = -1

    @contextmanager
    def TimePhase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    def WrapKernel(self, fn, name: str, *, group_by_decode_step: bool = False):
        """Return a wrapper that records every call's duration under `name`."""

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self.kernel_times[name].append(dt)
                if group_by_decode_step and self._decode_step_idx >= 0:
                    bucket = self.kernel_per_decode_step[name]
                    while len(bucket) <= self._decode_step_idx:
                        bucket.append([])
                    bucket[self._decode_step_idx].append(dt)

        return wrapper

    def BeginDecodeStep(self) -> None:
        self._decode_step_idx += 1

    @property
    def num_decode_steps(self) -> int:
        return self._decode_step_idx + 1


def _install_num_layers_override(engine: LLMEngine, n: int) -> None:
    """Monkey-patch the loader to truncate the model to N layers post-load.

    Validation-only knob. Must be installed BEFORE engine.init_model so the
    truncation lands before the executor compiles kernels for the wrong layer
    count (decode_stacked program is parameterised on num_layers).
    """
    if n <= 0:
        raise ValueError(f"--num-layers-override must be positive, got {n}")
    orig_load = engine._model_loader.load

    def load_truncated(*args, **kwargs):
        loaded = orig_load(*args, **kwargs)
        if n >= loaded.config.num_hidden_layers:
            return loaded
        new_config = dataclasses.replace(loaded.config, num_hidden_layers=n)
        loaded.runtime_model.layers = loaded.runtime_model.layers[:n]
        loaded.runtime_model.config = new_config
        print(
            f"[override] num_hidden_layers: "
            f"{loaded.config.num_hidden_layers} -> {n}",
            flush=True,
        )
        return LoadedModel(
            model_id=loaded.model_id,
            model_dir=loaded.model_dir,
            config=new_config,
            tokenizer=loaded.tokenizer,
            layer_specs=loaded.layer_specs[:n],
            runtime_model=loaded.runtime_model,
        )

    engine._model_loader.load = load_truncated


def InstallProfiling(engine: LLMEngine, model_id: str, collector: _TimingCollector) -> None:
    """Wrap executor.run_prefill / run_decode and the four compiled kernels so
    timings flow into `collector`. Must run AFTER engine.init_model().
    """
    executor = engine._executor  # type: ignore[attr-defined]
    compiled = executor._compiled[model_id]  # type: ignore[attr-defined]

    # Per-layer kernel wrappers. compiled.prefill / compiled.decode are invoked
    # once per transformer layer inside run_prefill / run_decode respectively
    # (baseline non-L3 path only).
    compiled.prefill = collector.WrapKernel(compiled.prefill, "kernel.prefill_layer")
    compiled.decode = collector.WrapKernel(
        compiled.decode, "kernel.decode_layer", group_by_decode_step=True
    )
    compiled.final_rms = collector.WrapKernel(compiled.final_rms, "kernel.final_rms")
    compiled.lm_head = collector.WrapKernel(compiled.lm_head, "kernel.lm_head")
    # L3 l3_generate: one dispatch drives prefill_all + decode_all (all-layers L2).
    if compiled.l3_generate is not None:
        compiled.l3_generate = collector.WrapKernel(
            compiled.l3_generate, "kernel.l3_generate", group_by_decode_step=True
        )

    # Top-level executor API wrappers.
    orig_prefill = executor.run_prefill
    orig_decode = executor.run_decode

    def timed_prefill(*args, **kwargs):
        with collector.TimePhase("api.run_prefill"):
            return orig_prefill(*args, **kwargs)

    def timed_decode(*args, **kwargs):
        collector.BeginDecodeStep()
        with collector.TimePhase("api.run_decode"):
            return orig_decode(*args, **kwargs)

    executor.run_prefill = timed_prefill
    executor.run_decode = timed_decode


def SummarizeTimes(times: list[float]) -> str:
    if not times:
        return "n=0"
    total_ms = sum(times) * 1000
    avg_ms = statistics.mean(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    return (
        f"n={len(times):4d}  total={total_ms:9.1f}ms  "
        f"avg={avg_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms"
    )


def PrintTimingReport(collector: _TimingCollector, num_tokens: int, verbose: bool) -> None:
    print("\n" + "=" * 70)
    print("=== Timing Report ===")
    print("=" * 70)

    # Level 1: phases
    init = collector.phases.get("init_model", 0.0)
    gen = collector.phases.get("generate_total", 0.0)
    print(f"[phase] init_model           : {init:8.2f}s   (weight load + kernel compile)")
    print(f"[phase] generate (e2e)       : {gen:8.2f}s   ({num_tokens} tokens generated)")
    if num_tokens > 0 and gen > 0:
        print(f"[phase] throughput (e2e)     : {num_tokens / gen:8.2f} tok/s")

    # Level 2: executor API
    api_prefill = collector.phases.get("api.run_prefill", 0.0)
    api_decode = collector.phases.get("api.run_decode", 0.0)
    n_decode = collector.num_decode_steps
    print(f"[api]   run_prefill          : {api_prefill:8.2f}s   (1 call, TTFT-ish)")
    if n_decode > 0:
        print(
            f"[api]   run_decode total     : {api_decode:8.2f}s   "
            f"({n_decode} steps, avg {api_decode / n_decode * 1000:7.1f} ms/step, "
            f"{n_decode / api_decode:6.2f} step/s)"
        )
    else:
        print(f"[api]   run_decode total     : {api_decode:8.2f}s   (0 steps)")

    # Level 3: per-kernel aggregate
    print()
    print("[kernel] aggregate over all invocations:")
    for kname in (
        "kernel.prefill_layer",
        "kernel.decode_layer",
        "kernel.l3_generate",
        "kernel.final_rms",
        "kernel.lm_head",
    ):
        print(f"  {kname:24s}: {SummarizeTimes(collector.kernel_times.get(kname, []))}")

    if not verbose:
        print(
            "\n(use --profile-verbose for per-layer prefill times and per-step decode "
            "layer breakdowns)"
        )
        return

    # Verbose: per-layer prefill times (single prefill call -> N layer kernels)
    prefill_layers = collector.kernel_times.get("kernel.prefill_layer", [])
    if prefill_layers:
        print("\n--- per-layer prefill kernel times (single run_prefill) ---")
        for layer_idx, t in enumerate(prefill_layers):
            print(f"  layer {layer_idx:02d}: {t * 1000:8.2f} ms")

    # Verbose: per-step summary for decode (each step has N layer kernel calls)
    per_step = collector.kernel_per_decode_step.get("kernel.decode_layer", [])
    if per_step:
        print("\n--- per-decode-step layer kernel breakdown ---")
        print(f"{'step':>5}  {'layers':>6}  {'total(ms)':>10}  {'avg/layer(ms)':>14}  "
              f"{'min(ms)':>9}  {'max(ms)':>9}")
        for step_idx, layer_times in enumerate(per_step):
            if not layer_times:
                continue
            total_ms = sum(layer_times) * 1000
            avg_ms = statistics.mean(layer_times) * 1000
            min_ms = min(layer_times) * 1000
            max_ms = max(layer_times) * 1000
            print(
                f"{step_idx:>5d}  {len(layer_times):>6d}  {total_ms:>10.2f}  "
                f"{avg_ms:>14.2f}  {min_ms:>9.2f}  {max_ms:>9.2f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Qwen3-14B generation with the bundled PyPTO kernels.")
    parser.add_argument("--model-dir", required=True, help="Local model directory, e.g. a Hugging Face snapshot.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default="qwen3-14b-local")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--save-kernels-dir", default=None)
    parser.add_argument(
        "--l3",
        action="store_true",
        dest="l3_mode",
        help="Enable L3 mode: prefill and decode are dispatched as a single "
             "all-layers L2 program inside a shared Worker(level=3) per "
             "generate call.",
    )
    parser.add_argument(
        "--num-layers-override",
        type=int,
        default=None,
        help="Validation knob: truncate the loaded model to N transformer "
             "layers before compile/dispatch. Used to reduce HBM footprint "
             "while validating the L3 stacked-decode path on Qwen3-14B "
             "(40 layers won't fit alongside 23 GiB of PTO2 shared memory).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print phase / executor-API / per-kernel timing summary at the end.",
    )
    parser.add_argument(
        "--profile-verbose",
        action="store_true",
        help="Implies --profile. Also dump per-layer prefill times and "
             "per-decode-step layer breakdowns.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    profile_enabled = args.profile or args.profile_verbose
    collector = _TimingCollector() if profile_enabled else None

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device_id,
        save_kernels_dir=args.save_kernels_dir,
        l3_mode=args.l3_mode,
        l3_trace=args.profile_verbose,
    )
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )

    if args.num_layers_override is not None:
        _install_num_layers_override(engine, args.num_layers_override)

    init_t0 = time.perf_counter()
    engine.init_model(
        model_id=args.model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=256,
            max_batch_size=16,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
    )
    if collector is not None:
        collector.phases["init_model"] = time.perf_counter() - init_t0
        # Profiling must be installed AFTER init_model: register_model populates
        # executor._compiled[model_id] with the four kernel callables we wrap.
        InstallProfiling(engine, args.model_id, collector)

    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
    )

    num_tokens = 0
    gen_t0 = time.perf_counter()
    if args.stream:
        text_parts: list[str] = []
        result = engine.generate(args.model_id, args.prompt, config)
        for chunk in result:
            text_parts.append(chunk)
            print(chunk, end="", flush=True)
        print()
        num_tokens = len(text_parts)
    else:
        result = engine.generate_result(args.model_id, args.prompt, config)
        num_tokens = len(result.token_ids)
        print(f"text: {result.text}")
        print(f"token_ids: {result.token_ids}")
        print(f"finish_reason: {result.finish_reason}")

    if collector is not None:
        collector.phases["generate_total"] = time.perf_counter() - gen_t0
        PrintTimingReport(collector, num_tokens=num_tokens, verbose=args.profile_verbose)


if __name__ == "__main__":
    main()
