"""End-to-end serving verification with real NPU executor (multiprocess worker)."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.async_engine import AsyncLLMEngine, ServingConfig
from core.tokenizer import TransformersTokenizerAdapter
from core.types import GenerateConfig, RuntimeConfig
from core.worker import WorkerConfig


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/data/linyifan/models/Qwen3-14B")
    parser.add_argument("--platform", type=str, default="a2a3")
    parser.add_argument("--device", "-d", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--in-process", action="store_true",
                        help="Run worker in-process (thread) instead of subprocess")
    parser.add_argument("--test", choices=["serving", "baseline", "all"], default="all",
                        help="Which test to run")
    return parser.parse_args()


async def test_serving(args, model_dir):
    """Test the async serving engine (multiprocess worker path)."""
    print(f"=== Serving V2 E2E Test ===")
    print(f"Mode: {'in-process' if args.in_process else 'multiprocess'}")
    print()

    print("[1/3] Loading tokenizer...")
    t0 = time.time()
    tokenizer = TransformersTokenizerAdapter.from_pretrained(model_dir)
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")

    print("[2/3] Creating AsyncLLMEngine + starting worker...")
    t1 = time.time()

    runtime_config = RuntimeConfig(
        page_size=256,
        max_batch_size=16,
        max_seq_len=512,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="float32",
        max_new_tokens=args.max_new_tokens,
    )

    worker_config = WorkerConfig(
        model_id="qwen3-14b",
        model_dir=model_dir,
        platform=args.platform,
        device_id=args.device,
        runtime_config=runtime_config,
        executor_cls="PyptoQwen14BExecutor",
    )

    serving_config = ServingConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=4096,
        long_prefill_token_threshold=2048,
        max_seq_len=512,
        block_size=256,
    )

    engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        in_process=args.in_process,
    )

    await engine.start()
    print(f"  Engine started in {time.time() - t1:.1f}s")

    print(f"[3/3] Testing single request (max_new_tokens={args.max_new_tokens})...")
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    t2 = time.time()
    full_text = ""
    finish_reason = ""
    token_count = 0
    async for output in engine.add_request("e2e-req-1", "What is 1+1?", config):
        if output.text:
            full_text = output.text
        if output.token_id is not None:
            token_count += 1
        if output.finished:
            finish_reason = output.finish_reason
            break
    elapsed = time.time() - t2

    print(f"  Response: {full_text[:100]}...")
    print(f"  Tokens: {token_count}, Time: {elapsed:.2f}s")
    print(f"  Finish reason: {finish_reason}")
    if token_count > 0:
        print(f"  Speed: {token_count/elapsed:.1f} tok/s")
    assert len(full_text) > 0 or token_count > 0, "No output generated"

    await engine.stop()
    print("  PASSED")


def test_baseline(args, model_dir):
    """Test the existing LLMEngine generate_batch path (L2 baseline)."""
    from core.engine import LLMEngine
    from core.kv_cache import KvCacheManager
    from core.pypto_executor import PyptoQwen14BExecutor

    print(f"=== L2 Baseline Generate Test ===")
    print()

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device,
    )
    engine = LLMEngine(kv_cache_manager=kv_cache_manager, executor=executor)

    runtime_config = RuntimeConfig(
        page_size=256,
        max_batch_size=16,
        max_seq_len=512,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="float32",
        max_new_tokens=8,
    )

    print("[1/2] Loading model...")
    t0 = time.time()
    engine.init_model("qwen3-14b", model_dir, runtime_config=runtime_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("[2/2] Running generate_batch (max_new_tokens=8)...")
    config = GenerateConfig(max_new_tokens=8, temperature=0.0)
    t1 = time.time()
    results = engine.generate_batch("qwen3-14b", ["What is 1+1?"], config)
    elapsed = time.time() - t1
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Result: {results[0].text}")
    print(f"  Token IDs: {results[0].token_ids}")
    print(f"  Finish reason: {results[0].finish_reason}")
    assert len(results[0].token_ids) > 0, "No tokens generated"
    print("  PASSED")


async def main():
    args = parse_args()
    model_dir = args.model_dir
    if not Path(model_dir).is_dir():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Model: {model_dir}")
    print(f"Platform: {args.platform}, Device: {args.device}")
    print()

    if args.test in ("baseline", "all"):
        test_baseline(args, model_dir)
        print()

    if args.test in ("serving", "all"):
        await test_serving(args, model_dir)
        print()

    print("=== All tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
