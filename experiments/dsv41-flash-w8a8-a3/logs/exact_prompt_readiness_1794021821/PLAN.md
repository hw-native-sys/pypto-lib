# Exact prompt length and strict DP agreement — next generate-only repair plan

This is READ-ONLY readiness. The current45 model sources remain frozen at
ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.
No model edit, NPU work or checkpoint payload read occurred. Main will not apply
changes until the reviewer seals/parks and root assigns the next bounded GO.
Existing repair authorization covers the proposed logic; no user permission
round is needed.

## Concrete compiler feasibility

The existing prefill_fwd._prefill_entry(tokens) factory caches by any supplied
Python integer; there is no bucket check inside that factory. The standalone
validation CLI has SUPPORTED_TILES=(2,16,128), and generate.main also has a
bucket guard. Those are different from a compiler/token-axis requirement.

Queued CPU task112436_323729726258 COMPLETEexit0 compiled the UNMODIFIED full40
real-expert entry from signatures at N=3 (odd),6 (even),17 (nonbucket). Each
normalized ONBOARD, emitted a2a3 metadata with w1 prefix[16,40,24], supplied ZERO
tensor samples and called no fixtures. Generated receive-X protocol buffers
bind the exact N:11,796,480 /23,592,960 /66,846,720B. One HOST chip call/rank
remains. compile_results.json and raw queue log bind the actual artifacts.
This proves compilation and specialization only: partial-row native arithmetic,
N=1/empty-compressed-prefix behavior, C2 odd tail state and real numerical
acceptance are not established by compile success.

## Exclusive next source scope

Main alone owns generate.py plus its own artifacts/docs/ledgers. No worker or
operator/forward/golden/loader edit is required for the proposed path. Existing
SWA/MLA/C2 transforms, quantization, ABI/topology, model thresholds, persistent
state helpers and source comparison semantics remain owned by their accepted
implementations. F09 general dynamic/idle/unequal batches remain separate.

1. Build/tokenize the requested prompt without altering its token IDs. In the
   device path derive local prefill_tokens=N=len(prompt['ids']) immediately.
   Reject an empty prompt explicitly. Use existing _prefill_entry(N),
   build_entry_contracts(N,decode_tokens=1,...) and exact-N inputs. Derive page
   capacity from start_position+N+requested output budget. Keep all N rows
   active; no inactive-padding contract is asserted.
2. Remove the generation path's call to fit_prompt_to_tile and its synthetic
   " Please continue." extension. Remove/restrict that helper if no other
   actual consumer remains. Do not simply bypass the extension while keeping a
   larger tile: token_counts, state/cache writes, row selection and handoff all
   must use N. Keep prefill sample rowN-1 and first decode position=start+N.
   Pass N to the handoff tile/length arguments and subsequent metadata logic.
3. Adapt only the generation CLI bucket restriction. Compile-only and standalone
   validation interfaces remain explicit about their concrete selected shapes.
   The proposed device path uses exact prompt length; report the actual token
   count and document --tokens as a compile/report setting rather than silently
   claiming it remains a device padding contract. Do not silently truncate a
   long prompt. Bound the initial path by its independently validated maximum
   (existing128-token prefix and actual fit budget); reject unsupported lengths
   with a clear count. The factory's8192 window capacity is not a fit or
   correctness promise for every N up to8192.
4. Preserve correct current token-history/EOS logic. No teacher forcing or
   golden logits enter sampling. Decode stays capacity1 and consumes the actual
   immediately preceding greedy token. Do not bundle Engram hashing, broad
   padding support, EOS rewrites or unrelated generator optimizations.

## Strict DP agreement with durable failure evidence

Keep existing TP4 global vocabulary reconstruction and deterministic tie
ordering. For check_groups=True, compute and retain all four DP groups' local
TP shard maxima, reconstructed global IDs, group greedy IDs and values. A
mismatch must raise a dedicated structured sampling failure rather than return
DP0's choice as a valid token.

Attach context before failure: phase(prefill/decode), selected row, current
position/input token, generated history so far, all four group results and the
exact conflicting IDs/scores. Record the structured evidence in the top-level
result/error payload and tee log even though device_run does not return.
Normal successful decode steps must also retain their agreement/group evidence;
currently that field is dropped. A failed selection must not append an accepted
output token or advance generation history. check_groups=False, if kept for
explicit diagnostic callers, must be clearly reported as unchecked.

This can stay within generate.py using a small exception/evidence record and
caller context. Do not convert disagreement into an incidental crash or an
acceptance-threshold change. Existing determinism.stable is informational; any
separate determinism gate change should remain explicitly scoped, not bundled
by inference.

## Meaningful next checks, not yet run

CPU host spies on a copied or later-assigned generate.py can use small synthetic
vocabulary shards (without tensors for model weights). Required controls:

- Original short IDs are byte-for-byte unchanged, no continuation encode call,
  exact-N factory/spec/active count, correct rowN-1/start+N/handoff metadata.
  Put poison argmax values in a hypothetical bucket tail and require the old
  appended/bucket behavior to fail the intended unmodified-prompt assertion.
- Independently compute four-shard global argmax for each of4 DP groups,
  including nonzero shard winners and ties. Positive agreement succeeds;
  disagreement in each nonzero DP group rejects and retains exact structured
  evidence, in both prefill and a later decode step.
- Verify actual previous selected token is fed at positions start+N,+1,+2,
  failures do not append a token, and the already correct first/later EOS cases
  stop at the same boundary. No original Engram history is introduced.
- Existing metadata/no-payload scalar/stream contract checks and source hash /
  meaningful negative controls remain. Native runtime tests must explicitly
  normalize ONBOARD and verify a2a3/real runtime logs.

After source GO, native bounded exact-N odd/even/ragged cases must validate all
19 source-defined full-forward outputs with unchanged comparators, actual
last-row logits and real returned cache/carry state. At least an odd prefix
ending on an even C2 position must feed its next odd decode through the carried
pair boundary. Preserve the independent128-prefix page/parity chain and stale
published-row control disposition; do not claim those from small exact-N
compiles or host spies. N=1 and any newly advertised range boundary need their
own validation or an explicit unsupported contract. Real weight payloads remain
held until independent functional and measured fit gates pass.

## Isolated draft value and current stop

While the reviewer completes long frozen-prefix checks, an ISOLATED copy /
patch for generate.py plus CPU spies would add value: it can reproduce and
reject old prompt mutation and DP-disagreement behavior without touching the
root45-source snapshot or running NPU work. It is one small host integration
change, not a new framework; no implementing worker is needed. Root may assign
that bounded artifact-only draft separately. Current authorization is readiness
only, so no such draft or source patch was created in this unit.

Main parks with this compiler evidence and concrete ownership/test plan. The
current model freeze and full-weight guard remain unchanged.
