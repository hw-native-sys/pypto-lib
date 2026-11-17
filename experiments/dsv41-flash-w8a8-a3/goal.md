# Experiment goal (draft; remaining agreement pending)

Owner: the launched model Main Agent for this experiment (the Claude Code
session that created this directory). The harness's native-goal step maps to
this host's goal facility and has not been activated yet; nothing below is
an activated goal.

## Requested objective

For the read-only checkpoint `DeepSeek-V4.1-Flash-w8a8`, under the
vllm-ascend A3-supported form as the spec basis, deliver accepted modules
and complete-model compiled prefill/decode forwards on the agreed Ascend
(`a2a3`, 16 logical devices) topology, plus reproducible real-checkpoint
text generation through those entries, with engineering and
execution-structure acceptance (including the tokenwise full-stack
negative control) executed and closed by the owner. See
`HARNESS_ROOT/prompts/PROMPT.md` and ACCEPTANCE.md.

## Agreed decisions (user, 2026-11-02; spec revision 1)

1. Checkpoint: `/mnt/data/models/DeepSeek-V4.1-Flash-w8a8` (read-only;
   completeness verified, see WEIGHT_MANIFEST.md).
2. Environment: `.venv-local` via `activate-pypto.sh` (verified; see
   ENVIRONMENT.md rev 1).
3. Spec basis: the vllm-ascend A3-supported form (tutorial rev vLLM Ascend
   v0.30.0rc2, section 5.1 single-node colocated: DP4/TP4 + EP on 16 logical
   devices, W8A8 + INT8 Engram host offload, block 128, max-model-len
   150000, max-num-batched-tokens 8192, max-num-seqs 32).
4. Goal framing: FRESH implementation for A3 with this checkpoint's W8A8
   policy. The existing `models/deepseek_v4_1_flash` targets the A5 platform
   with a different quantization policy; it is a semantics/decomposition
   reference only, not reusable implementation.
5. Topology: A3 form — DP4/TP4 + expert parallel on 16 logical devices;
   pypto platform `-p a2a3`. Measured feasibility passes (34.63 GiB/rank
   weights vs 57.6 GiB usable; see MODEL_SPEC.md).
6. Modality: text-only first phase; vision tower and aligner excluded.
7. DSpark and MTP layers excluded from the first goal (checkpoint carries
   them; out of scope). Engram (layers 1/14) is in scope, host-offloaded,
   int8_sym group-32 + fp32 scales (file headers are ground truth; the
   checkpoint README's decoded-BF16 prose is stale).
8. Validation policy (user, 2026-11-03, after the machine switch): from now
   on, run validation on real NPU devices via the host's task-submit queue
   (skill `task-submit-npu`); avoid simulator validation. Pure-host checks
   (weight-loader sampling, tokenizer) stay host-side. Prior simulator
   evidence remains historical; device gates were required regardless.
9. Scope revision 2 (user, 2026-11-03): implement REAL dynamic batch
   (multi-request, per-request block-table cache addressing) and REAL
   dynamic S (runtime token counts and context lengths inside one compiled
   artifact — dynamic axes and runtime scalars, not per-shape
   recompilation), with the serving-adapted KV cache format: 128-block
   paged raw window cache, paged compressed cache, index cache,
   per-request index block tables, request ids, compressed lens/slots —
   the A5 interface shape. Chain and entry interface parameters follow the
   A5 implementation as closely as possible; divergences are permitted only
   for A3-specific quantization (per-channel int8 + offsets + dynamic
   per-token int8) and parallelism (DP4/TP4 + EP16 exchange) choices, plus
   individually justified and recorded exceptions (known candidate: how
   staged-validation hooks survive the port-level signature). Multi-request
   batching is IN SCOPE as of this revision, superseding the harness-default
   single-request scope (which was never explicitly user-ratified).

## Remaining minor items (not implementation-blocking)

Chat-template proposal for generation fixtures (plain completion +
deepseek_v41 format) and numerical thresholds — both resolved at fixture
calibration on the original HF reference. Implementation is delegated to
the launched model Main Agent; the outer session acts as launcher/observer
and collects time statistics in TIME_LOG.md.

## Decision 10 — temporary first-generation Engram scope (user, 2026-11-06)

User: "engram 可以暂时不用，或者放在host计算和查表".
For the first real-checkpoint prefill/continuous-decode generation milestone,
Engram may be explicitly DISABLED. Outputs, commands, fixtures and reports must
state that scope; they do not establish equivalence to the original full-Engram
model. Preserve the implemented gate/order/loader repairs. Original Engram
hash/history/table computation may be restored on the host in a later unit;
its missing autoregressive hashing is no longer a blocker for this temporary
Engram-disabled first path. Decision7 remains the eventual Engram scope unless
the user revises it again. Decision9 dynamic-support requirements remain open;
a bounded capacity1 first stream cannot silently close them. No relaxation of
real-NPU evidence, cache/state correctness, checkpoint quantization or topology.


## Decision 11 — configurable A3 EP/TP through real generation (user, 2026-11-08)

User: "把 EP/TP 改为可配置的形态，参考 A5 实现的方案，验通吐字为止".
This explicitly supersedes the fixed-topology restriction in decisions3/5 for
the new configurable-parallelism work. Use the A5 import-time CLI --tp/--ep
configuration approach as a semantic/interface reference, preserving A3 W8A8
quantization and the existing paged/dynamic/cache/shared-weight contracts.
Derive and validate world/DP/rank groups and all affected sharding/collectives,
loader/metadata/goldens/generation/sampling/admission paths; do not expose flags
while leaving internal fixed16/4 behavior. Preserve the prior TP4/EP16 default
compatibility unless a recorded correctness reason requires otherwise.
Declare actual valid combinations and fail invalid/nondivisible/unfit cases
early. Resource-constrained smaller EP may use meaningful real-NPU component
validation; do not claim full-model generation fits from synthetic evidence.
The new milestone MUST reach real-checkpoint text generation on an explicitly
selected nondefault valid EP/TP configuration, beyond source/compile-only work.
Select that configuration from measured resources without another routine user
permission request. Engram may remain disabled under decision10; no MTP/DSpark
or A5-specific quantization migration is authorized. Old source/results/failures
remain immutable historical evidence. Native Main/workers and independent
reviewers execute work/validation; observer coordinates, records dated directives
and time, and persists until the requested configurable path generates text.


## Decision11 bounded outcome — 2026-11-08

Configurable A3 EP/TP implementation and the requested actual-text milestone are accepted within recorded bounds. Configuration exposes TP1/2/4/8 and EP2/4/8/16 subject to EP>=TP and divisibility (13 pairs), WORLD=EP and DP=EP//TP, defaultTP4/EP16; A5 is the configuration-interface reference, with A3 W8A8 and replicated32-head indexer retained. FullE40 EP8 and smaller do not fit the current whole-expert resident layout; their component proof is separate from full-model fit.

Independent original-golden native checks passed for TP8 local8-head attention (three cases), EP2/TP1 MoE (three durable distinct cases, exactcounts12/6/6 and idle-rank service), and TP2/EP4 reduction including257-column tail. Current sourcef0e91a30 has freshE40prefill/decode signatures and177 independent source/resource/admission/PREQUEUE checks. Actual task130915_150556113554 ran fullreal W8A8 TP8/EP16/DP2 with Engram disabled, promptOnceupona, generated IDs[1014,14,1031,515,1613,1], visibletextOnceuponatime,therewas..., then genuineEOS after5decodecalls. One613256211456B canonical upload/912bases and decodezero newstaticIO/upload/allocation are independently verified.

Actual post-audit4713/163217 completed0 with4363PASS/5stricttelemetryUNMET and bounded_unmetEMPTY. Root disposition is logs/root_configurable_real_generation_1794111133_disposition.json. The wholecase remains NOT_ACCEPTED: nativeexit1/modelCLI0/outerFAIL259noncriticalsamplinggaps/PSSnull; knownpartial12,300,867,584B is not a complete peak. Oldfailures remain immutable. Original numerical/quant parity, quality, all-topology fullmodel/native support, generalserving, Engram/MTP and complete-memory/processhistory claims remain open. Rawterminaltime and late progressnotification are recorded separately with unknown communication-gap cause, without invented monitoring. Owner/audit final seals and latest recovery status are linked in TASKS/COORDINATION_STATE. No further model implementation or NPU action follows from this closeout.


## Decision12 — multi-prompt quality diagnosis, primary TP4 (user, 2026-11-09)

User asks: "之前吐字是不是还有问题？你能用别的prompt试一试吗？如果有问题定位一下" and then steers "主要检查 TP4/DP4/EP16 这个配置吧". Primary new real-checkpoint experiment is TP4/DP4/EP16 on the current A3 W8A8 implementation. Run several meaningfully different prompts, evaluate response correctness/relevance and EOS behavior, and localize any observed issue with concrete controls; text emission/finite votes/CLI0/EOS alone are not quality acceptance. TP8 is an optional diagnostic control only if justified, not the main matrix or an automatic second weight-load run. Choose representative factual/arithmetic/English/Chinese completion and appropriate chat-format controls using CPU-verified tokenization. Preserve per-case prompt/IDs/template/outputs/termination/history/DP votes and avoid teacher forcing or forced EOS.

Prefer a single resident static-weight upload for several independently initialized cases; any host-driver/admission extension must keep fresh per-case KV/protocol state and exact source/Root/resource/accounting guards. Prior N3/cap16 single-prompt receipts are historical scopes, not permission for new prompts. User-authorized native Main/workers implement necessary bounded host integration and independent reviewers execute validation; observer coordinates without implementing or personally running tests. Engram may stay disabled per Decision10, explicitly qualified; its absence cannot be asserted as the cause of bad text without evidence. No MTP/DSpark, A5 quantization migration, relaxed numeric criteria, old-evidence rewrites or dependency/admin changes. Full original-model parity may remain unavailable, and any such diagnosis limit must be stated.
