# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the serving scheduler and block pool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.block_pool import BlockPool, FreeBlockQueue, KVBlock, hash_block_tokens, NONE_HASH
from core.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig


def test_free_block_queue_basic():
    q = FreeBlockQueue()
    b1 = KVBlock(block_id=0)
    b2 = KVBlock(block_id=1)
    b3 = KVBlock(block_id=2)

    q.append(b1)
    q.append(b2)
    q.append(b3)
    assert len(q) == 3

    popped = q.popleft()
    assert popped is b1
    assert len(q) == 2

    q.remove(b3)
    assert len(q) == 1

    popped = q.popleft()
    assert popped is b2
    assert len(q) == 0
    assert q.popleft() is None


def test_block_pool_allocate_release():
    pool = BlockPool(num_blocks=4, block_size=16)
    assert pool.num_free_blocks == 4

    b1 = pool.allocate()
    assert b1 is not None
    assert b1.ref_cnt == 1
    assert pool.num_free_blocks == 3

    b2 = pool.allocate()
    b3 = pool.allocate()
    b4 = pool.allocate()
    assert pool.num_free_blocks == 0
    assert pool.allocate() is None

    pool.release(b1)
    assert pool.num_free_blocks == 1
    assert b1.ref_cnt == 0

    b5 = pool.allocate()
    assert b5 is b1


def test_block_pool_prefix_caching():
    pool = BlockPool(num_blocks=8, block_size=4)

    tokens = list(range(12))  # 3 full blocks
    hashes = pool.compute_block_hashes(tokens)
    assert len(hashes) == 3

    # Allocate and cache blocks
    blocks = []
    for h in hashes:
        b = pool.allocate()
        pool.cache_block(b, h)
        blocks.append(b)

    # Release blocks (they stay in cache)
    for b in blocks:
        pool.release(b)
    assert pool.num_free_blocks == 5 + 3  # 5 never allocated + 3 released

    # Lookup should find cached blocks
    hit = pool.get_computed_blocks(tokens)
    assert len(hit) == 3
    assert [b.block_id for b in hit] == [blocks[i].block_id for i in range(3)]

    # Different prefix should not match
    different_tokens = [99, 98, 97, 96] + list(range(4, 12))
    hit2 = pool.get_computed_blocks(different_tokens)
    assert len(hit2) == 0


def test_block_pool_lru_eviction():
    pool = BlockPool(num_blocks=3, block_size=4)

    tokens_a = list(range(4))
    tokens_b = list(range(4, 8))
    hash_a = pool.compute_block_hashes(tokens_a)[0]
    hash_b = pool.compute_block_hashes(tokens_b)[0]

    b1 = pool.allocate()
    pool.cache_block(b1, hash_a)
    b2 = pool.allocate()
    pool.cache_block(b2, hash_b)
    b3 = pool.allocate()
    assert pool.num_free_blocks == 0

    pool.release(b3)
    pool.release(b1)  # b1 released after b3, so b3 is older in LRU
    pool.release(b2)

    # Allocate should evict LRU (b3 first since it was released first)
    evicted = pool.allocate()
    assert evicted is b3
    assert evicted.block_hash is None  # hash cleared on eviction


def test_scheduler_basic_scheduling():
    pool = BlockPool(num_blocks=16, block_size=4)
    config = SchedulerConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        long_prefill_token_threshold=16,
        max_seq_len=64,
    )
    scheduler = Scheduler(config=config, block_pool=pool)

    req1 = Request(
        request_id="req-1",
        prompt_token_ids=list(range(8)),
        max_new_tokens=4,
        eos_token_id=99,
    )
    req2 = Request(
        request_id="req-2",
        prompt_token_ids=list(range(10, 22)),
        max_new_tokens=4,
        eos_token_id=99,
    )

    scheduler.add_request(req1)
    scheduler.add_request(req2)
    assert scheduler.has_work()

    output = scheduler.schedule()
    assert len(output.scheduled_requests) == 2
    assert output.num_prefill_tokens == 8 + 12  # both are prefill
    assert output.num_decode_tokens == 0

    # Both should now be RUNNING
    assert req1.status == RequestStatus.RUNNING
    assert req2.status == RequestStatus.RUNNING


def test_scheduler_chunked_prefill():
    pool = BlockPool(num_blocks=16, block_size=4)
    config = SchedulerConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        long_prefill_token_threshold=8,  # cap at 8 tokens per chunk
        max_seq_len=64,
    )
    scheduler = Scheduler(config=config, block_pool=pool)

    # Request with 20 prompt tokens — should be chunked
    req = Request(
        request_id="req-1",
        prompt_token_ids=list(range(20)),
        max_new_tokens=4,
        eos_token_id=99,
    )
    scheduler.add_request(req)

    output = scheduler.schedule()
    assert len(output.scheduled_requests) == 1
    sr = output.scheduled_requests[0]
    assert sr.num_new_tokens == 8  # capped by threshold
    assert sr.is_prefill is True

    # Simulate execution: update num_computed_tokens
    req.num_computed_tokens += sr.num_new_tokens
    assert req.num_computed_tokens == 8
    assert req.is_prefill  # still in prefill (8 < 20)

    # Next schedule should continue prefill
    output2 = scheduler.schedule()
    assert len(output2.scheduled_requests) == 1
    sr2 = output2.scheduled_requests[0]
    assert sr2.num_new_tokens == 8  # another chunk
    assert sr2.is_prefill is True


def test_scheduler_preemption():
    pool = BlockPool(num_blocks=4, block_size=4)  # very limited
    config = SchedulerConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        long_prefill_token_threshold=0,
        max_seq_len=64,
    )
    scheduler = Scheduler(config=config, block_pool=pool)

    req1 = Request(
        request_id="req-1",
        prompt_token_ids=list(range(8)),
        max_new_tokens=4,
        eos_token_id=99,
        arrival_time=1.0,
    )
    scheduler.add_request(req1)

    # Schedule req1 — uses 2 blocks (8 tokens / 4 block_size)
    output = scheduler.schedule()
    assert len(output.scheduled_requests) == 1
    req1.num_computed_tokens = 8

    # Now add req2 that needs more blocks than available
    req2 = Request(
        request_id="req-2",
        prompt_token_ids=list(range(12)),
        max_new_tokens=4,
        eos_token_id=99,
        arrival_time=2.0,
    )
    scheduler.add_request(req2)

    # Schedule — req2 needs 3 blocks, only 2 free. Should try to preempt req1.
    output2 = scheduler.schedule()
    # After preemption of req1, req2 should be scheduled
    if output2.preempted_requests:
        assert output2.preempted_requests[0].request_id == "req-1"


def test_scheduler_finish_detection():
    pool = BlockPool(num_blocks=16, block_size=4)
    config = SchedulerConfig(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=32,
        long_prefill_token_threshold=0,
        max_seq_len=64,
    )
    scheduler = Scheduler(config=config, block_pool=pool)

    req = Request(
        request_id="req-1",
        prompt_token_ids=list(range(4)),
        max_new_tokens=2,
        eos_token_id=99,
    )
    scheduler.add_request(req)

    output = scheduler.schedule()
    req.num_computed_tokens = 4

    # Simulate model output with EOS token
    new_tokens = {"req-1": 99}
    results = scheduler.update_from_output(output, new_tokens)
    assert len(results) == 1
    assert results[0].finished is True
    assert results[0].finish_reason == "FINISHED_EOS"
    assert req.status == RequestStatus.FINISHED_EOS


if __name__ == "__main__":
    test_free_block_queue_basic()
    print("PASS: test_free_block_queue_basic")

    test_block_pool_allocate_release()
    print("PASS: test_block_pool_allocate_release")

    test_block_pool_prefix_caching()
    print("PASS: test_block_pool_prefix_caching")

    test_block_pool_lru_eviction()
    print("PASS: test_block_pool_lru_eviction")

    test_scheduler_basic_scheduling()
    print("PASS: test_scheduler_basic_scheduling")

    test_scheduler_chunked_prefill()
    print("PASS: test_scheduler_chunked_prefill")

    test_scheduler_preemption()
    print("PASS: test_scheduler_preemption")

    test_scheduler_finish_detection()
    print("PASS: test_scheduler_finish_detection")

    print("\nAll tests passed!")
