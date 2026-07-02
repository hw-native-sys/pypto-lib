# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level checks for Qwen3 greedy sampling kernels."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "models" / "qwen3" / "14b"


def _source(name: str) -> str:
    return (QWEN / name).read_text(encoding="utf-8")


def test_fixed_batch_greedy_tie_break_reads_winning_chunk_from_local_slice() -> None:
    source = _source("decode_fwd.py")

    assert "winning_logits = pl.slice(logits" in source
    assert "winning_logits, [0, pl.cast(scan_t, pl.INDEX)]" in source
    assert "val = pl.read(logits, [b, token_idx])" not in source
    assert "chunk_base_idx = pl.cast(chunk_base, target_type=pl.INDEX)" in source
    assert "local_scores = pl.fillpad" not in source
    assert "winning_logits = pl.fillpad" not in source
    assert "REAL_VOCAB_TAIL" in source
    assert "fillpad" in source
    assert "SAMPLE_VOCAB_CHUNK" in source


def test_prefill_keeps_sampling_in_standalone_kernels() -> None:
    source = _source("prefill_fwd.py")

    assert "_greedy_sample_inline" not in source
    assert "_token_embed_inline" not in source
    assert "sampled_ids:" not in source
    assert "embed_weight:" not in source
    assert "winning_logits = pl.slice(logits" not in source


def test_decode_comment_matches_next_hidden_seed() -> None:
    source = _source("decode_fwd.py")

    assert "loop-carried `cur` is seeded from next_hidden" in source
    assert "loop-carried `cur` is seeded from hidden_states" not in source
