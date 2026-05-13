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
import contextlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

GenerateConfig = None
LLMEngine = None
RuntimeConfig = None
KvCacheManager = None
PyptoQwen14BExecutor = None


_VALID_BACKENDS = {"cpu", "npu"}
_L3_BATCH_TILE = 16
_EXIT_COMMANDS = {"/exit", "/quit"}
_HELP_COMMANDS = {"/help", "?"}
_CONFIG_COMMANDS = {"/config"}
_CLEAR_COMMANDS = {"/clear"}


@dataclass(frozen=True)
class ModelCliConfig:
    model_id: str
    model_dir: str
    model_format: str
    loader_options: dict[str, object]


@dataclass(frozen=True)
class NpuCliConfig:
    platform: str = "a2a3"
    device_id: int = 0
    save_kernels_dir: str | None = None
    pypto_root: str | None = None
    l3_mode: bool = False


@dataclass(frozen=True)
class ServingConfig:
    model: ModelCliConfig
    runtime: RuntimeConfig
    generation: GenerateConfig
    backend: str
    npu: NpuCliConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypto-serving",
        description="Run local interactive generation with the llm module.",
    )
    parser.add_argument("--config", required=True, help="Path to the user-written JSON config file.")
    parser.add_argument("--prompt", help="Prompt text for one-shot generation.")
    parser.add_argument("--interactive", action="store_true", help="Read prompts interactively after loading the model.")
    parser.add_argument("--serve", action="store_true", help="Start HTTP serving mode with OpenAI-compatible API.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the serving server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port for the serving server (default: 8000).")
    parser.add_argument("--stream", action="store_true", help="Override generation.stream=true for this run.")
    parser.add_argument("--l3", action="store_true", help="Override npu.l3=true for this run.")
    parser.add_argument(
        "--max-num-running-reqs",
        type=int,
        default=32,
        help="Max concurrent requests in serving mode (default: 32).",
    )
    parser.add_argument(
        "--max-num-scheduled-tokens",
        type=int,
        default=4096,
        help="Max tokens scheduled per iteration in serving mode (default: 4096).",
    )
    parser.add_argument(
        "--long-prefill-token-threshold",
        type=int,
        default=2048,
        help="Chunked prefill threshold in serving mode (default: 2048).",
    )
    parser.add_argument(
        "--show-startup-logs",
        action="store_true",
        help="Show model loading and kernel compilation logs. Startup logs are suppressed by default.",
    )
    return parser


def load_serving_config(
    config_path: str | Path,
    *,
    stream_override: bool = False,
    l3_override: bool = False,
) -> ServingConfig:
    _ensure_core_imports()
    path = Path(config_path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read config {path}: {exc}") from exc

    root = _require_mapping(raw, "config")
    model_section = _require_mapping(root.get("model"), "model")
    runtime_section = _require_mapping(root.get("runtime"), "runtime")
    generation_section = _require_mapping(root.get("generation"), "generation")
    npu_section = _optional_mapping(root.get("npu"), "npu")

    backend = _get_str(runtime_section, "backend", "npu").lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"runtime.backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}")

    model_dir = _get_required_str(model_section, "model_dir", "model")
    model = ModelCliConfig(
        model_id=_get_str(model_section, "model_id", "qwen3-14b-local"),
        model_dir=model_dir,
        model_format=_get_str(model_section, "model_format", "huggingface"),
        loader_options=dict(_optional_mapping(model_section.get("loader_options"), "model.loader_options")),
    )

    generation_defaults = GenerateConfig()
    generation = GenerateConfig(
        max_new_tokens=_get_int(generation_section, "max_new_tokens", generation_defaults.max_new_tokens),
        temperature=_get_float(generation_section, "temperature", generation_defaults.temperature),
        top_p=_get_float(generation_section, "top_p", generation_defaults.top_p),
        top_k=_get_optional_int(generation_section, "top_k"),
        stop=_get_stop(generation_section),
        stream=stream_override or _get_bool(generation_section, "stream", generation_defaults.stream),
    )

    npu_l3_mode = l3_override or _get_bool_alias(npu_section, ("l3", "l3_mode"), False)

    page_size = _get_optional_int(runtime_section, "page_size")
    if page_size is None:
        page_size = 256 if backend == "npu" else 64
    max_batch_size_default = _L3_BATCH_TILE if backend == "npu" and npu_l3_mode else 1
    max_batch_size = _get_int(runtime_section, "max_batch_size", max_batch_size_default)
    if backend == "npu" and npu_l3_mode and max_batch_size != _L3_BATCH_TILE:
        raise ValueError(
            f"npu.l3 requires runtime.max_batch_size={_L3_BATCH_TILE}; "
            f"got {max_batch_size}."
        )
    runtime = RuntimeConfig(
        page_size=page_size,
        max_batch_size=max_batch_size,
        max_seq_len=_get_int(runtime_section, "max_seq_len", 4096),
        device=_get_str(runtime_section, "device", "cpu"),
        kv_dtype=_get_str(runtime_section, "kv_dtype", "bfloat16"),
        weight_dtype=_get_str(runtime_section, "weight_dtype", "float32"),
        total_kv_pages=_get_optional_int(runtime_section, "total_kv_pages"),
        max_new_tokens=generation.max_new_tokens,
    )

    npu = NpuCliConfig(
        platform=_get_str(npu_section, "platform", "a2a3"),
        device_id=_get_int(npu_section, "device_id", 0),
        save_kernels_dir=_get_optional_str(npu_section, "save_kernels_dir"),
        pypto_root=_get_optional_str(npu_section, "pypto_root"),
        l3_mode=npu_l3_mode,
    )

    return ServingConfig(
        model=model,
        runtime=runtime,
        generation=generation,
        backend=backend,
        npu=npu,
    )


def create_engine(config: ServingConfig) -> LLMEngine:
    _ensure_core_imports(executor=config.backend == "npu")
    if config.backend == "cpu":
        return LLMEngine()

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        pypto_root=config.npu.pypto_root,
        platform=config.npu.platform,
        device_id=config.npu.device_id,
        save_kernels_dir=config.npu.save_kernels_dir,
        l3_mode=config.npu.l3_mode,
    )
    return LLMEngine(kv_cache_manager=kv_cache_manager, executor=executor)


def init_engine(engine: LLMEngine, config: ServingConfig) -> None:
    model_dir = Path(config.model.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    engine.init_model(
        model_id=config.model.model_id,
        model_dir=str(model_dir),
        model_format=config.model.model_format,
        runtime_config=config.runtime,
        **config.model.loader_options,
    )


def generate_once(
    engine: LLMEngine,
    config: ServingConfig,
    prompt: str,
    *,
    stdout: TextIO | None = None,
    show_role: bool = False,
) -> None:
    stdout = stdout or sys.stdout
    if show_role:
        print("[assistant]", file=stdout)
    if config.generation.stream:
        result = engine.generate(config.model.model_id, prompt, config.generation)
        for chunk in _as_iterator(result):
            print(chunk, end="", flush=True, file=stdout)
        print(file=stdout)
        return

    result = engine.generate_result(config.model.model_id, prompt, config.generation)
    print(f"text: {result.text}", file=stdout)
    print(f"token_ids: {result.token_ids}", file=stdout)
    print(f"finish_reason: {result.finish_reason}", file=stdout)


def run_interactive(
    engine: LLMEngine,
    config: ServingConfig,
    *,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO | None = None,
) -> None:
    stdout = stdout or sys.stdout
    _print_interactive_banner(config, stdout)
    while True:
        try:
            prompt = input_fn("[user] ")
        except EOFError:
            print(file=stdout)
            return
        stripped = prompt.strip()
        if not stripped:
            continue
        if stripped in _EXIT_COMMANDS:
            print("Bye.", file=stdout)
            return
        if stripped in _HELP_COMMANDS:
            _print_interactive_help(stdout)
            continue
        if stripped in _CONFIG_COMMANDS:
            _print_runtime_summary(config, stdout)
            continue
        if stripped in _CLEAR_COMMANDS:
            print("--- new prompt session ---", file=stdout)
            continue
        generate_once(engine, config, prompt, stdout=stdout, show_role=True)


def run_serve(
    engine: LLMEngine,
    config: ServingConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_num_running_reqs: int = 32,
    max_num_scheduled_tokens: int = 4096,
    long_prefill_token_threshold: int = 2048,
) -> None:
    import asyncio

    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("Serving mode requires uvicorn. Install with: pip install uvicorn") from e

    from ..core.async_engine import AsyncLLMEngine, ServingConfig as AsyncServingConfig
    from ..core.server import create_serving_app
    from ..core.tokenizer import TransformersTokenizerAdapter
    from ..core.worker import WorkerConfig

    model_id = config.model.model_id
    model_dir = str(Path(config.model.model_dir).resolve())

    # Load only the tokenizer in main process (lightweight)
    tokenizer = TransformersTokenizerAdapter.from_pretrained(model_dir)

    # Worker config — worker process loads model independently
    executor_kwargs = {}
    if config.npu.pypto_root:
        executor_kwargs["pypto_root"] = config.npu.pypto_root
    if config.npu.save_kernels_dir:
        executor_kwargs["save_kernels_dir"] = config.npu.save_kernels_dir
    if config.npu.l3_mode:
        executor_kwargs["l3_mode"] = config.npu.l3_mode

    worker_config = WorkerConfig(
        model_id=model_id,
        model_dir=model_dir,
        platform=config.npu.platform,
        device_id=config.npu.device_id,
        runtime_config=config.runtime,
        executor_cls="PyptoQwen14BExecutor" if config.backend == "npu" else "ModelExecutor",
        executor_kwargs=executor_kwargs,
    )

    serving_config = AsyncServingConfig(
        max_num_running_reqs=max_num_running_reqs,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_seq_len=config.runtime.max_seq_len,
        block_size=config.runtime.page_size,
    )

    async_engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    app = create_serving_app(async_engine, model_id)

    @app.on_event("startup")
    async def startup():
        await async_engine.start()

    @app.on_event("shutdown")
    async def shutdown():
        await async_engine.stop()

    print(f"Starting PyPTO serving on {host}:{port}")
    print(f"  Model: {model_id} (loaded in worker process)")
    print(f"  Platform: {config.npu.platform}, Device: {config.npu.device_id}")
    print(f"  Max running requests: {max_num_running_reqs}")
    print(f"  Max scheduled tokens/iter: {max_num_scheduled_tokens}")
    print(f"  Chunked prefill threshold: {long_prefill_token_threshold}")
    print(f"  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health")

    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = sum([bool(args.prompt), bool(args.interactive), bool(args.serve)])
    if modes != 1:
        raise SystemExit("Specify exactly one of --prompt, --interactive, or --serve.")

    with _startup_log_context(enabled=not args.show_startup_logs):
        config = load_serving_config(
            args.config,
            stream_override=args.stream,
            l3_override=args.l3,
        )
        if not args.serve:
            # Only load model in main process for non-serve modes.
            # In serve mode, the worker process loads the model independently.
            engine = create_engine(config)
            init_engine(engine, config)
        else:
            engine = None

    if args.serve:
        run_serve(
            engine,
            config,
            host=args.host,
            port=args.port,
            max_num_running_reqs=args.max_num_running_reqs,
            max_num_scheduled_tokens=args.max_num_scheduled_tokens,
            long_prefill_token_threshold=args.long_prefill_token_threshold,
        )
    elif args.interactive:
        run_interactive(engine, config)
    else:
        generate_once(engine, config, args.prompt)
    return 0


def _ensure_core_imports(*, executor: bool = False) -> None:
    global GenerateConfig, LLMEngine, RuntimeConfig, KvCacheManager, PyptoQwen14BExecutor

    if GenerateConfig is None or LLMEngine is None or RuntimeConfig is None:
        try:
            from ..core import GenerateConfig as imported_generate_config
            from ..core import LLMEngine as imported_engine
            from ..core import RuntimeConfig as imported_runtime_config
        except ImportError:
            from core import GenerateConfig as imported_generate_config
            from core import LLMEngine as imported_engine
            from core import RuntimeConfig as imported_runtime_config

        if GenerateConfig is None:
            GenerateConfig = imported_generate_config
        if LLMEngine is None:
            LLMEngine = imported_engine
        if RuntimeConfig is None:
            RuntimeConfig = imported_runtime_config

    if KvCacheManager is None:
        try:
            from ..core.kv_cache import KvCacheManager as imported_kv_cache_manager
        except ImportError:
            from core.kv_cache import KvCacheManager as imported_kv_cache_manager
        KvCacheManager = imported_kv_cache_manager

    if executor and PyptoQwen14BExecutor is None:
        try:
            from ..core.pypto_executor import PyptoQwen14BExecutor as imported_executor
        except ImportError:
            from core.pypto_executor import PyptoQwen14BExecutor as imported_executor
        PyptoQwen14BExecutor = imported_executor


def _print_interactive_banner(config: ServingConfig, stdout: TextIO) -> None:
    print("PyPTO Serving interactive generation", file=stdout)
    _print_runtime_summary(config, stdout)
    print("Commands: /help, /config, /clear, /exit, /quit", file=stdout)
    print("Enter a prompt to generate text.", file=stdout)


def _print_interactive_help(stdout: TextIO) -> None:
    print("Commands:", file=stdout)
    print("  /help    Show this help.", file=stdout)
    print("  /config  Show active model, backend, runtime, and generation settings.", file=stdout)
    print("  /clear   Print a separator for the next prompt.", file=stdout)
    print("  /exit    Exit interactive mode.", file=stdout)
    print("  /quit    Exit interactive mode.", file=stdout)


def _print_runtime_summary(config: ServingConfig, stdout: TextIO) -> None:
    print(
        "Config: "
        f"model_id={config.model.model_id}, "
        f"backend={config.backend}, "
        f"max_seq_len={config.runtime.max_seq_len}, "
        f"max_new_tokens={config.generation.max_new_tokens}, "
        f"temperature={config.generation.temperature}, "
        f"top_p={config.generation.top_p}, "
        f"top_k={config.generation.top_k}, "
        f"stream={config.generation.stream}",
        file=stdout,
    )
    if config.backend == "npu":
        print(
            f"NPU: platform={config.npu.platform}, device_id={config.npu.device_id}, "
            f"l3={config.npu.l3_mode}",
            file=stdout,
        )


@contextlib.contextmanager
def _startup_log_context(*, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    old_log_level = os.environ.get("PTO_LOG_LEVEL")
    os.environ.setdefault("PTO_LOG_LEVEL", "error")
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        if old_log_level is None:
            os.environ.pop("PTO_LOG_LEVEL", None)
        else:
            os.environ["PTO_LOG_LEVEL"] = old_log_level


def _as_iterator(result: str | Iterator[str]) -> Iterator[str]:
    if isinstance(result, str):
        yield result
        return
    yield from result


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _optional_mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _get_required_str(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{section_name}.{key} must be a non-empty string.")
    return value


def _get_str(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _get_optional_str(section: Mapping[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string when provided.")
    return value


def _get_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _get_optional_int(section: Mapping[str, Any], key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when provided.")
    return value


def _get_float(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _get_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _get_bool_alias(section: Mapping[str, Any], keys: tuple[str, ...], default: bool) -> bool:
    for key in keys:
        if key in section:
            return _get_bool(section, key, default)
    return default


def _get_stop(section: Mapping[str, Any]) -> tuple[str, ...]:
    value = section.get("stop", ())
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError("stop must be a string or a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("stop must contain only strings.")
    return tuple(value)


if __name__ == "__main__":
    raise SystemExit(main())
