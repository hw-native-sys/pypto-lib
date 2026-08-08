# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent per-request state for fused DeepSeek-V4 MTP decode.

The scheduler batch is transient: a request may occupy a different local row
on every step.  The tensors in this module form a persistent device-side pool
instead.  Serving passes a small descriptor, ``(slot_id, generation)``, for
each current batch row so the fused kernel can:

1. load the previous tail/draft into the main-model decode inputs;
2. run main decode, MTP verification, and the next-draft model; and
3. commit the resulting tail/draft back to the same persistent slot.

For the supported ``S == 2`` layout, flattened rows ``2*r`` and ``2*r+1``
belong to local batch row ``r``.
"""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, DECODE_TOKENS
from lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD


B = DECODE_BATCH
S = DECODE_SEQ
T = DECODE_TOKENS

# ``state_meta`` is a persistent pool indexed by serving-assigned state slot,
# not by the request's transient row in the current decode batch.
# ``valid`` means that the slot contains a completely initialized payload.  In
# the current FIFO lifecycle it changes to 1 on first assignment and is not
# cleared on release: serving stops publishing that slot, and the generation is
# incremented before reuse.  It is therefore an initialization guard, not the
# scheduler's live-request/ownership bit.
STATE_VALID = 0
# Allocation tag (ABA guard).  Serving increments it whenever a freed slot is
# assigned to a new request; stale prepared metadata must match neither the new
# request's tokens nor its tail state.
STATE_GENERATION = 1
STATE_TAIL_POSITION = 2
STATE_COMMITTED_COUNT = 3
STATE_META_WIDTH = 4

STATE_TAIL_TOKEN = 0
STATE_DRAFT_TOKEN = 1
STATE_TOKEN_WIDTH = 2

assert S == 2, "persistent MTP state requires decode_seq=2"


@pl.jit.inline
def prepare_decode_from_device_state(
    state_slot_ids: pl.Tensor[[B], pl.INT32],
    state_generations: pl.Tensor[[B], pl.INT32],
    state_tokens: pl.Tensor[[B, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[B, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    tail_token_ids: pl.Tensor[[B], pl.INT64],
    tail_positions: pl.Tensor[[B], pl.INT32],
):
    """Late-bind recurrent decode fields from stable device slots.

    This runs at the beginning of the fused decode, before the main model.  For
    each active local batch row ``r``, it validates the prepared descriptor and
    rewrites the placeholder host inputs from the authoritative device state::

        row0 = 2*r: input=tail,  position=tail_position + 1
        row1 = 2*r+1: input=draft, position=tail_position + 2
        kv_seq_lens[r] = tail_position + 3

    Positions are zero-based.  Therefore ``tail_position + 3`` is the KV
    length after the two rows at positions ``tail_position+1`` and
    ``tail_position+2`` have been included.

    Args:
        state_slot_ids: ``[B]`` transient-row to persistent-slot mapping.
            ``-1`` marks an inactive/padded row.
        state_generations: ``[B]`` allocation generations captured by serving
            when it prepared the batch.  This must match the slot metadata.
        state_tokens: Persistent ``[B, 2]`` pool indexed by slot.  Column 0 is
            the last committed tail token; column 1 is the draft to verify.
        state_meta: Persistent ``[B, 4]`` pool indexed by slot, containing
            ``valid``, ``generation``, ``tail_position``, and
            ``committed_count``.
        input_ids: In/out flattened main-decode tokens ``[T=B*S]``.  Valid
            request rows are overwritten with ``[tail, draft]``.
        position_ids: In/out flattened main-decode positions ``[T]``.
        kv_seq_lens: In/out main-model KV lengths ``[B]``.
        tail_token_ids: In/out ``[B]`` verifier scratch.  It receives the old
            committed tail token used when the draft is rejected.
        tail_positions: In/out ``[B]`` verifier scratch.  It receives the old
            committed tail position used when the draft is rejected.

    A row is consumed only when ``slot_id >= 0``, the slot is valid, and its
    generation matches.  Otherwise every caller-provided padded value is left
    untouched.  Returned tensors alias the in-place-updated arguments and make
    the mutation/dependency explicit to PyPTO's dataflow compiler.
    """
    # One core owns these tightly packed scalar updates.  The metadata volume
    # is tiny, and single ownership avoids adjacent scalar DMA write races.
    for core in pl.spmd(1, name_hint="mtp_state_prepare"):
        for request in pl.range(core, B):
            slot_raw = pl.read(state_slot_ids, [request])
            if slot_raw >= 0:
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    row0 = request * S
                    row1 = row0 + 1
                    tail_token = pl.read(state_tokens, [slot, STATE_TAIL_TOKEN])
                    draft_token = pl.read(state_tokens, [slot, STATE_DRAFT_TOKEN])
                    tail_position = pl.read(state_meta, [slot, STATE_TAIL_POSITION])
                    pl.write(input_ids, [row0], tail_token)
                    pl.write(input_ids, [row1], draft_token)
                    pl.write(
                        position_ids,
                        [row0],
                        pl.cast(tail_position + 1, target_type=pl.INT32),
                    )
                    pl.write(
                        position_ids,
                        [row1],
                        pl.cast(tail_position + 2, target_type=pl.INT32),
                    )
                    pl.write(
                        kv_seq_lens,
                        [request],
                        pl.cast(tail_position + 3, target_type=pl.INT32),
                    )
                    pl.write(tail_token_ids, [request], tail_token)
                    pl.write(tail_positions, [request], tail_position)
    return input_ids, position_ids, kv_seq_lens, tail_token_ids, tail_positions


@pl.jit.inline
def advance_decode_device_state(
    state_slot_ids: pl.Tensor[[B], pl.INT32],
    state_generations: pl.Tensor[[B], pl.INT32],
    state_tokens: pl.Tensor[[B, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[B, STATE_META_WIDTH], pl.INT32],
    committed_input_ids: pl.Tensor[[T], pl.INT64],
    committed_position_ids: pl.Tensor[[T], pl.INT32],
    next_sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    accepted_counts: pl.Tensor[[B], pl.INT32],
):
    """Commit verifier and draft-model outputs into their stable slots.

    This runs at the end of the fused decode.  ``verify_and_pack_mtp_tokens``
    has already normalized both acceptance outcomes into a two-row committed
    window for request ``r``::

        draft accepted (accepted=2): [main0, main1]
        draft rejected (accepted=1): [old_tail, main0]

    In both cases row ``2*r+1`` is consequently the newest committed tail.
    This helper stores that row as the next step's tail, stores the MTP model's
    sampled token as the next draft, advances the tail position, and increments
    the running committed-token count by ``accepted``.

    Args:
        state_slot_ids: ``[B]`` transient-row to persistent-slot mapping used
            for this invocation; ``-1`` means inactive.
        state_generations: ``[B]`` expected allocation generations captured
            with the prepared batch.
        state_tokens: In/out persistent ``[B, 2]`` pool.  Column 0 receives
            the newest committed tail and column 1 receives the next draft.
        state_meta: In/out persistent ``[B, 4]`` pool.  This function updates
            ``tail_position`` and ``committed_count`` only.
        committed_input_ids: ``[T]`` two-row committed windows produced by
            verification, not the original speculative main inputs.
        committed_position_ids: ``[T]`` positions aligned with those committed
            windows.
        next_sampled_ids: ``[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD]`` MTP LM-head
            sampling output.  Row ``r``, column 0 is request ``r``'s next draft.
        accepted_counts: ``[B]`` verifier result, currently 1 on rejection and
            2 when the single draft is accepted.

    The same valid-bit and generation check as the prepare side protects this
    writeback from a stale batch after request release and slot reuse.  Returns
    alias the two in-place-updated persistent pools.
    """
    # Keep the state transition single-owner for the same scalar-write reason
    # as the prepare helper above.
    for core in pl.spmd(1, name_hint="mtp_state_advance"):
        for request in pl.range(core, B):
            slot_raw = pl.read(state_slot_ids, [request])
            if slot_raw >= 0:
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    # Verification always packs the newest committed token and
                    # position into the second row, regardless of acceptance.
                    row1 = request * S + 1
                    accepted = pl.read(accepted_counts, [request])
                    next_draft = pl.cast(
                        pl.read(next_sampled_ids, [request, 0]),
                        target_type=pl.INT64,
                    )
                    pl.write(
                        state_tokens,
                        [slot, STATE_TAIL_TOKEN],
                        pl.read(committed_input_ids, [row1]),
                    )
                    pl.write(state_tokens, [slot, STATE_DRAFT_TOKEN], next_draft)
                    pl.write(
                        state_meta,
                        [slot, STATE_TAIL_POSITION],
                        pl.read(committed_position_ids, [row1]),
                    )
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED_COUNT])
                    pl.write(
                        state_meta,
                        [slot, STATE_COMMITTED_COUNT],
                        committed + accepted,
                    )
    return state_tokens, state_meta
