# First-text source sanity check — read-only

Scope: frozen generate.py SHA26b6db8451afa4488bff71699b94b7ffcf86abcec08e2919912e388fa904c8fd,
default noEngram single-request decode1 path, no decode-state override. This is
source inspection only: no added tests, NPU, payload reads or source edits.

## Row selection and padding

- generate.py1106 fit_prompt_to_tile extends a short tokenized prompt with
  repetitions of encode(" Please continue.") until its length equals the
  compiled tile. It does not pad that effective prompt. device_run invokes
  this before building inputs (2283) and samples len(prompt['ids'])-1 (2331).
  Thus it chooses the last valid row of the EFFECTIVE extended prompt.
- Concrete semantic limitation: that effective prompt differs from the user's
  original short prompt. The returned prompt record retains original fields
  and adds tile_extended_from; ids contain the added tokens. This is not
  ordinary short-prompt/padding support. An original prompt longer than the
  tile errors rather than chunking.
- build_prefill_inputs1025 can pad a shorter supplied record, but its plan and
  token_counts cover the entire tile (1007), so pad rows would enter cache /
  compressor state. prefill_tile_gap1132 and handoff_to_decode normally reject
  a shorter prompt; allowing padded prefill is explicitly diagnostic. Removing
  the extension alone would not establish correct active-length/state handoff.
- The current allactive128 native prefix cannot validate arbitrary short
  prompts or inactive padding. First normal-text scope must disclose the actual
  extended prompt or separately repair/accept short-prompt active-length and
  cache semantics. No such repair is made here.

## TP and DP greedy behavior

- _shard_argmax1756 obtains the local argmax and reconstructs global vocabulary
  ID as (rank %4)*32320+local_index. greedy_sample1761 compares the four TP
  shard maxima within one DP group. Four contiguous TP4 groups cover ranks
  0..3,4..7,8..11,12..15. These offsets match129280 global vocabulary entries.
- Exact ties retain the first local max and first TP candidate, yielding the
  lowest global ID among tied shard maxima under the current ordering.
- Concrete gap: check_groups=True only records other DP groups' differing token
  IDs; it does NOT reject disagreement. The driver always uses default DP0.
  First-token records retain dp_disagreements, but subsequent step dictionaries
  copy only token_id/logit and do not retain that disagreement field. Neither
  check_groups nor the recorded determinism.stable flag acts as an acceptance
  gate in device_run. A future real-text acceptance must inspect/gate those
  results explicitly rather than assume consistency because the flag is true.

## Actual logits, next token, history and EOS

- Sampling reads prefill_values['logits'] AFTER the real dispatch. Repeated
  prefill runs reset fresh nonstatic inputs and each sample uses the actual
  resulting logits; the final repetition supplies the next token. No golden
  logits or fixture token list supplies generation choices.
- First decode position is start_position+len(effective prompt ids) (2377).
  current is the actual selected prefill token. advance_decode_inputs1094 fills
  that token into the one-row input on every rank, sets active count1, rebuilds
  slots/lens/RoPE for that position and keeps persistent caches.
- Each decode dispatch updates the actual logits tensor before greedy_sample;
  decode_rows is1 by the enforced decode contract, so row decode_rows-1 is0.
  It does not sample a repeated/padded multirow decode tail. The sampled output
  becomes current for the following step; position increments exactly once.
- generated starts with the actual prefill selection and appends each actual
  decode selection. tokenizer.decode(generated) produces displayed history.
  noEngram means no original Engram hashing/history/compression on this path;
  original-model parity is explicitly false.
- An initial EOS returns before decode preparation. Later EOS is appended to
  history and stops the loop before being fed back as a new model input.
  The last position record describes the consumed input, not an extra EOS row.
- For max_new_tokens=1 and a non-EOS first token, the current source still
  prepares/uploads the decode phase before an empty decode loop. This is an
  extra phase memory/loading cost, not a row-selection defect. It must be
  accounted when planning a one-token real-checkpoint smoke run.

## Disposition

No source change or native semantic claim. Immediate future acceptance should
state the effective full-tile prompt, require DP agreement/determinism results
rather than treating recorded flags as gates, and validate actual returned
three-step state separately. Short original-prompt semantics and general
padding remain a separate concrete gap. Current source freeze and full-weight
hold remain in force; root decides any repair scope after independent evidence.
