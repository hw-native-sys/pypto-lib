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
import sys
from pathlib import Path


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device_id,
        save_kernels_dir=args.save_kernels_dir,
    )
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )
    engine.init_model(
        model_id=args.model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=256,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
    )
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
    )
    if args.stream:
        result = engine.generate(args.model_id, args.prompt, config)
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        result = engine.generate_result(args.model_id, args.prompt, config)
        print(f"text: {result.text}")
        print(f"token_ids: {result.token_ids}")
        print(f"finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    main()
