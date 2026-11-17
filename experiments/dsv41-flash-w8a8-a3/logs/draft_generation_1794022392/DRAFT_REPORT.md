# Isolated exact-prompt / strict-DP draft — ROOT NOT MODIFIED

Only draft_model/generate.py is edited. It starts from frozen generate.py
26b6db8451afa4488bff71699b94b7ffcf86abcec08e2919912e388fa904c8fd.
The other44 copied model Python files are read-only and hash-identical support
files. ROOT45 sources remain frozen at ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.
draft.patch is a reviewable one-file patch for a LATER apply assignment. It has
not been applied, imported by production, or used for native acceptance.

## Draft behavior

- Original prompt IDs are preserved. Device generation and its host preflight
  derive exact N and call the existing prefill factory/contracts with N. There
  is no "Please continue" augmentation or inactive-padding workaround. All N
  rows are active, sample rowN-1 and handoff tile/length match N, and decode
  starts at start_position+N. The artificial CLI bucket check remains for
  compile-only; generation uses original prompt length. Help describes --tokens
  as a compile/report setting. Standalone forward validation interfaces remain
  unchanged.
- The bounded first path explicitly supports2..128 original prompt tokens;
  empty,1-token and longer inputs reject unchanged. N=1 is deliberately not
  advertised before its empty-compressed-prefix boundary has native evidence.
  The8192 fixed window capacity is not asserted as a correctness/fit range.
- Requested DP agreement now fails on any differing greedy ID among4 DP
  replicas. Structured votes retain each group's four TP maxima/global IDs /
  scores, selected result, phase, row, prompt IDs, consumed position/input token
  and history so far. Every successful decode step retains the sampling record.
  Main result/error JSON and log preserve failure evidence for both prefill and
  later decode. A rejected selection is not appended to accepted history.
- Existing TP4 global vocabulary reconstruction and tie order remain. Explicit
  diagnostic check_groups=False returns an unchecked label. Existing actual
  prior-logit feedback and EOS behavior remain; no Engram history or new model
  arithmetic is introduced.
- The full-weight guard is HARD and AST-identical to the base, including the
  --allow-full-materialize case. PreparedDispatch is also AST-identical. There
  is no real-payload, fit-allocation or root-apply authorization in this draft.

AST changes are limited to build_parser/device_run/dry_run/greedy_sample/main,
new SamplingDisagreement/exact_prompt_tokens, and removed prompt-augmentation
helper/import. No operator, loader, golden, comparator or runtime source change.

## CPU host-spy evidence only

Tasks113956_344002713949 and114217_40248837890 ran queued --no-device. The first
passed all host spies but then failed ruff on the obsolete encode import after
removing augmentation. That raw log is preserved. The unused import was removed;
114217 completedexit0, all spies and ruff PASS.

The test imports the isolated draft and frozen base under different module
names. It replaces tokenizer/entries/weight owner/dispatch with tiny CPU spies;
no model arithmetic, native backend or weights execute. The requested-mode CLI
failure records contain an explicit CPUfake test_scope tag and MUST NOT be
read as NPU evidence.

- N3/6/17 retain original IDs, use exact factory/contracts and row, pass exact
  handoff N, then feed actual selected tokens6/9/11 at consecutive positions.
  For N3/start125, decode inputs are6@128,9@129,11@130. This is host control flow,
  not native cache/page correctness.
- Old baseline appends3 tokens to16 and selects poison-tail ID15 instead of6;
  its first decode position is141 instead of128. This rejects the actual old
  prompt mutation, not just a source-string assertion.
- Old baseline returns a DP0 token despite another group's different winner.
  The draft rejects6 group/phase combinations (groups1/2/3, prefill/decode),
  retaining full votes/context. Actual main error-handler paths preserve JSON
  for prefill and second decode; the latter history stays[6,9], input9.
- Independent concatenated-shard argmax and nonzero-shard tie cases pass.
  Initial and later EOS stop at the existing boundaries. Unsupported0/1/129
  lengths reject without changing IDs; opt-in and default full-weight guard
  calls both still reject. ROOT45 and44 support copies remain equal.

Native exact-N tail arithmetic, odd C2 carry,19 decode /17 prefill owning gates,
real DP replicas and actual returned-state handoff remain UNVALIDATED by this
CPU draft. Prior zero-sample N3/6/17 compilation is separate readiness evidence.
No new source, checkpoint payload, NPU, large allocation or production apply
was authorized or performed here.

## Parallel checkpoint preparation

The first checkpoint_pipeline prototype is sealed separately and has11 toy
CPU rejection controls; it is explicitly PARTIAL with16MiB cap and no production
resume. Root resumed the same worker in disjoint checkpoint_pipeline_v2 to wire
actual frozen helper/model adapters. That scope remains separate from this
one-file draft. Main will inspect and integrate its report without changing
ROOT or the active reviewer helper. Both unconditioned and staged references,
17 prefill /19 decode keys and unchanged comparison gates must be preserved.
