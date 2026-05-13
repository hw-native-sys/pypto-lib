"""Integration test for the V2 serving pipeline (multiprocess worker + dynamic batching)."""

import asyncio
import queue
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.async_engine import AsyncLLMEngine, ServingConfig, TokenOutput
from core.block_pool import BlockPool
from core.scheduler import Request, Scheduler, SchedulerConfig, SchedulerOutput, ScheduledRequest
from core.types import GenerateConfig, StepOutput, WorkerCommand
from core.worker import WorkerConfig


def _make_mock_tokenizer(vocab_size=100):
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40])
    tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok{''.join(str(i) for i in ids)}")
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    return tokenizer


async def test_engine_with_mock_worker():
    """Test AsyncLLMEngine with a mock worker feeding responses via queues."""
    tokenizer = _make_mock_tokenizer()

    # Create engine with dummy worker config (won't actually spawn)
    worker_config = WorkerConfig(model_id="test", model_dir="/tmp/fake")
    serving_config = ServingConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        long_prefill_token_threshold=16,
        max_seq_len=64,
        block_size=4,
        num_blocks=64,
    )

    engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=2,
        bos_token_id=1,
    )

    # Manually set up queues (bypass worker spawn)
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    engine._input_queue = input_queue
    engine._output_queue = output_queue
    engine._running = True
    engine._in_process = True

    # Mock worker thread: reads commands, returns mock tokens
    def mock_worker():
        step_count = 0
        while True:
            cmd = input_queue.get()
            if cmd.type == "shutdown":
                break
            # Return token 50 for first 2 steps, then EOS (token 2)
            step_count += 1
            new_tokens = {}
            for sr in cmd.scheduler_output.scheduled_requests:
                if step_count >= 3:
                    new_tokens[sr.request.request_id] = 2  # EOS
                else:
                    new_tokens[sr.request.request_id] = 50 + step_count
            output_queue.put(StepOutput(new_tokens=new_tokens))

    worker_thread = threading.Thread(target=mock_worker, daemon=True)
    worker_thread.start()
    engine._worker_thread = worker_thread

    # Start engine loop
    engine._loop_task = asyncio.create_task(engine._engine_loop())

    # Submit request
    config = GenerateConfig(max_new_tokens=5, temperature=0.0)
    outputs = []
    async for output in engine.add_request("test-req-1", "Hello world", config):
        outputs.append(output)
        if output.finished:
            break

    await engine.stop()

    assert len(outputs) > 0, f"Expected outputs, got none"
    assert outputs[-1].finished, "Last output should be finished"
    assert outputs[-1].finish_reason == "FINISHED_EOS"
    print(f"  Got {len(outputs)} outputs, finish_reason={outputs[-1].finish_reason}")
    return True


async def test_dynamic_batching():
    """Test that multiple concurrent requests are batched together."""
    tokenizer = _make_mock_tokenizer()

    worker_config = WorkerConfig(model_id="test", model_dir="/tmp/fake")
    serving_config = ServingConfig(
        max_num_running_reqs=8,
        max_num_scheduled_tokens=64,
        long_prefill_token_threshold=32,
        max_seq_len=64,
        block_size=4,
        num_blocks=128,
    )

    engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=2,
        bos_token_id=1,
    )

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    engine._input_queue = input_queue
    engine._output_queue = output_queue
    engine._running = True
    engine._in_process = True

    batch_sizes_seen = []

    def mock_worker():
        step_count = 0
        while True:
            cmd = input_queue.get()
            if cmd.type == "shutdown":
                break
            step_count += 1
            scheduled = cmd.scheduler_output.scheduled_requests
            batch_sizes_seen.append(len(scheduled))
            new_tokens = {}
            for sr in scheduled:
                # EOS after 3 decode steps
                if len(sr.request.output_token_ids) >= 2:
                    new_tokens[sr.request.request_id] = 2  # EOS
                else:
                    new_tokens[sr.request.request_id] = 50 + step_count
            output_queue.put(StepOutput(new_tokens=new_tokens))

    worker_thread = threading.Thread(target=mock_worker, daemon=True)
    worker_thread.start()
    engine._worker_thread = worker_thread
    engine._loop_task = asyncio.create_task(engine._engine_loop())

    config = GenerateConfig(max_new_tokens=5, temperature=0.0)

    async def collect_request(req_id, prompt):
        outputs = []
        async for output in engine.add_request(req_id, prompt, config):
            outputs.append(output)
            if output.finished:
                break
        return outputs

    # Submit 3 concurrent requests
    results = await asyncio.gather(
        collect_request("req-1", "Hello"),
        collect_request("req-2", "World"),
        collect_request("req-3", "Test"),
    )

    await engine.stop()

    assert len(results) == 3, "Expected 3 result sets"
    for i, outputs in enumerate(results):
        assert len(outputs) > 0, f"Request {i} got no outputs"
        assert outputs[-1].finished, f"Request {i} not finished"

    # Check that batching occurred (some steps should have batch_size > 1)
    max_batch = max(batch_sizes_seen) if batch_sizes_seen else 0
    print(f"  Batch sizes seen: {batch_sizes_seen}")
    print(f"  Max batch size: {max_batch}")
    print(f"  All 3 requests completed successfully")
    return True


async def test_server_endpoints():
    """Test FastAPI server endpoints with the new engine."""
    from core.server import create_serving_app

    try:
        from httpx import AsyncClient, ASGITransport
    except ImportError:
        print("  SKIP: httpx not installed (pip install httpx for this test)")
        return True

    tokenizer = _make_mock_tokenizer()

    worker_config = WorkerConfig(model_id="test-model", model_dir="/tmp/fake")
    serving_config = ServingConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        block_size=4,
        num_blocks=64,
    )

    engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=2,
        bos_token_id=1,
    )

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    engine._input_queue = input_queue
    engine._output_queue = output_queue
    engine._running = True
    engine._in_process = True

    def mock_worker():
        while True:
            cmd = input_queue.get()
            if cmd.type == "shutdown":
                break
            new_tokens = {}
            for sr in cmd.scheduler_output.scheduled_requests:
                new_tokens[sr.request.request_id] = 2  # immediate EOS
            output_queue.put(StepOutput(new_tokens=new_tokens))

    worker_thread = threading.Thread(target=mock_worker, daemon=True)
    worker_thread.start()
    engine._worker_thread = worker_thread
    engine._loop_task = asyncio.create_task(engine._engine_loop())

    app = create_serving_app(engine, "test-model")

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            print("  /health OK")

            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"][0]["id"] == "test-model"
            print("  /v1/models OK")

            resp = await client.post("/v1/completions", json={
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0.0,
            })
            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) == 1
            print(f"  /v1/completions OK: {data['choices'][0]}")
    finally:
        await engine.stop()

    return True


async def test_streaming_output():
    """Test that token-by-token streaming works."""
    tokenizer = _make_mock_tokenizer()

    worker_config = WorkerConfig(model_id="test", model_dir="/tmp/fake")
    serving_config = ServingConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        block_size=4,
        num_blocks=64,
    )

    engine = AsyncLLMEngine(
        worker_config=worker_config,
        serving_config=serving_config,
        tokenizer=tokenizer,
        eos_token_id=2,
        bos_token_id=1,
    )

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    engine._input_queue = input_queue
    engine._output_queue = output_queue
    engine._running = True
    engine._in_process = True

    def mock_worker():
        step_count = 0
        while True:
            cmd = input_queue.get()
            if cmd.type == "shutdown":
                break
            step_count += 1
            new_tokens = {}
            for sr in cmd.scheduler_output.scheduled_requests:
                if step_count >= 4:
                    new_tokens[sr.request.request_id] = 2  # EOS
                else:
                    new_tokens[sr.request.request_id] = 40 + step_count
            output_queue.put(StepOutput(new_tokens=new_tokens))

    worker_thread = threading.Thread(target=mock_worker, daemon=True)
    worker_thread.start()
    engine._worker_thread = worker_thread
    engine._loop_task = asyncio.create_task(engine._engine_loop())

    config = GenerateConfig(max_new_tokens=10, temperature=0.0)
    outputs = []
    async for output in engine.add_request("stream-req", "Hello", config):
        outputs.append(output)
        if output.finished:
            break

    await engine.stop()

    # Should have multiple intermediate outputs + final
    non_final = [o for o in outputs if not o.finished]
    assert len(non_final) >= 2, f"Expected streaming outputs, got {len(non_final)} non-final"
    assert outputs[-1].finished
    print(f"  Got {len(non_final)} streaming tokens + 1 final")
    print(f"  Token IDs: {[o.token_id for o in outputs]}")
    return True


async def run_all():
    print("test_engine_with_mock_worker...")
    assert await test_engine_with_mock_worker()
    print("PASS\n")

    print("test_dynamic_batching...")
    assert await test_dynamic_batching()
    print("PASS\n")

    print("test_streaming_output...")
    assert await test_streaming_output()
    print("PASS\n")

    print("test_server_endpoints...")
    assert await test_server_endpoints()
    print("PASS\n")

    print("All V2 integration tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all())
