"""Quick comparison: existing engine vs async engine decode."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/data/linyifan/models/Qwen3-14B")
    parser.add_argument("--platform", type=str, default="a2a3")
    parser.add_argument("--device", "-d", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    from core.engine import LLMEngine
    from core.kv_cache import KvCacheManager
    from core.pypto_executor import PyptoQwen14BExecutor
    from core.types import GenerateConfig, RuntimeConfig

    print(f"=== Existing Engine Baseline Test ===")
    print(f"Platform: {args.platform}, Device: {args.device}")

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device,
    )
    engine = LLMEngine(kv_cache_manager=kv_cache_manager, executor=executor)

    model_id = "qwen3-14b"
    runtime_config = RuntimeConfig(
        page_size=256,
        max_batch_size=16,
        max_seq_len=512,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="float32",
        max_new_tokens=8,
    )

    print("[1] Loading model...")
    t0 = time.time()
    engine.init_model(model_id, args.model_dir, runtime_config=runtime_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("[2] Running generate_batch (existing engine, max_new_tokens=8)...")
    config = GenerateConfig(max_new_tokens=8, temperature=0.0)
    t1 = time.time()
    results = engine.generate_batch(model_id, ["What is 1+1?"], config)
    elapsed = time.time() - t1
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Result: {results[0].text}")
    print(f"  Token IDs: {results[0].token_ids}")
    print(f"  Finish reason: {results[0].finish_reason}")
    print()
    print("=== Baseline test complete ===")


if __name__ == "__main__":
    main()
