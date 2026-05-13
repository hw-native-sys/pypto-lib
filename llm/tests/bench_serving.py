"""Serving benchmark: measures TTFT (prefill), per-token decode latency, and throughput."""

import argparse
import asyncio
import json
import time

import aiohttp


async def send_request_streaming(session, url, prompt, max_tokens, temperature):
    """Send a streaming request, measure TTFT and per-token latency."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    full_text = ""

    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if delta:
                full_text += delta
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)

    t_end = time.perf_counter()
    total = t_end - t0

    # Per-token decode intervals (between consecutive tokens, excluding first)
    decode_intervals = []
    for i in range(1, len(token_times)):
        decode_intervals.append(token_times[i] - token_times[i - 1])

    return {
        "total": total,
        "ttft": ttft or total,
        "decode_intervals": decode_intervals,
        "num_tokens": len(token_times),
        "text": full_text,
    }


async def send_request_non_streaming(session, url, prompt, max_tokens, temperature):
    """Send a non-streaming request, measure end-to-end latency."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
    elapsed = time.perf_counter() - t0
    text = result.get("choices", [{}])[0].get("text", "")
    return {
        "total": elapsed,
        "ttft": None,
        "decode_intervals": [],
        "num_tokens": 0,
        "text": text,
    }


def percentile(sorted_list, p):
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * p)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


async def run_bench(args):
    url = f"http://{args.host}:{args.port}/v1/completions"
    prompts = [
        "The capital of France is",
        "Explain quantum computing in one sentence:",
        "Write a haiku about the ocean:",
        "What is 2+2? Answer:",
        "The meaning of life is",
        "Python is a programming language that",
        "In the year 2050,",
        "The fastest animal on Earth is",
    ]

    tasks = []
    for i in range(args.num_requests):
        prompt = prompts[i % len(prompts)]
        tasks.append(prompt)

    print(f"=== PyPTO Serving Benchmark ===")
    print(f"Target: {url}")
    print(f"Requests: {args.num_requests}, Concurrency: {args.concurrency}")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print(f"Mode: {'streaming (TTFT + decode)' if args.stream else 'non-streaming (e2e only)'}")
    print()

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_request(session, prompt):
        async with sem:
            if args.stream:
                return await send_request_streaming(session, url, prompt, args.max_tokens, args.temperature)
            else:
                return await send_request_non_streaming(session, url, prompt, args.max_tokens, args.temperature)

    t_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        coros = [bounded_request(session, prompt) for prompt in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
    t_total = time.perf_counter() - t_start

    errors = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]

    if not successes:
        print(f"All {len(errors)} requests failed!")
        if errors:
            print(f"  First error: {errors[0]}")
        return

    # --- Aggregate metrics ---
    total_latencies = sorted(r["total"] for r in successes)
    total_tokens = sum(r["num_tokens"] for r in successes)
    throughput_req = len(successes) / t_total
    throughput_tok = total_tokens / t_total if total_tokens > 0 else 0

    print(f"--- End-to-End ---")
    print(f"Total time:      {t_total:.2f}s")
    print(f"Successful:      {len(successes)}/{args.num_requests}")
    print(f"Errors:          {len(errors)}")
    print(f"Throughput:      {throughput_req:.2f} req/s | {throughput_tok:.1f} tok/s")
    print(f"Latency avg:     {sum(total_latencies)/len(total_latencies):.3f}s")
    print(f"Latency p50:     {percentile(total_latencies, 0.5):.3f}s")
    print(f"Latency p99:     {percentile(total_latencies, 0.99):.3f}s")

    if args.stream:
        # TTFT (Time To First Token) ~ prefill latency
        ttfts = sorted(r["ttft"] for r in successes)
        print()
        print(f"--- TTFT (≈ prefill) ---")
        print(f"TTFT avg:        {sum(ttfts)/len(ttfts)*1000:.1f}ms")
        print(f"TTFT p50:        {percentile(ttfts, 0.5)*1000:.1f}ms")
        print(f"TTFT p99:        {percentile(ttfts, 0.99)*1000:.1f}ms")

        # Per-token decode latency (inter-token interval)
        all_intervals = []
        for r in successes:
            all_intervals.extend(r["decode_intervals"])
        if all_intervals:
            all_intervals.sort()
            print()
            print(f"--- Decode (per-token interval) ---")
            print(f"Tokens total:    {total_tokens}")
            print(f"Interval avg:    {sum(all_intervals)/len(all_intervals)*1000:.1f}ms")
            print(f"Interval p50:    {percentile(all_intervals, 0.5)*1000:.1f}ms")
            print(f"Interval p99:    {percentile(all_intervals, 0.99)*1000:.1f}ms")

    print()
    print(f"--- Sample outputs ---")
    for r in successes[:3]:
        print(f"  [total={r['total']:.2f}s tokens={r['num_tokens']}] {r['text'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyPTO serving")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-requests", "-n", type=int, default=8)
    parser.add_argument("--concurrency", "-c", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming to measure TTFT and decode latency")
    args = parser.parse_args()
    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
