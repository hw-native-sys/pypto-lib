# Independent logic review — frozen A3 W8A8 implementation

**Verdict: the snapshot does not implement the agreed model semantics.** There
are verified mathematical defects in both kernels and goldens, independently of
the outstanding decode state/pool publication failures. Fixing those publication
failures alone would not make this model equivalent to the agreed reference.
Generation also has source-level failures before and during decode.

This is a read-only review of the 45 Python files in [source](source), not a
review or acceptance of the concurrent repair tree. No model, golden, test,
threshold, checkpoint, PyPTO, TASKS, or shared environment was changed. No
subagents, NPU jobs, simulator jobs, compilation, commits, external issues, or
messages to other agents were used. Only this directory and append-only
`TIME_LOG.md` review-agent rows are written.

## Identity, authority, and reference provenance

- First measured start: **1793929688**, 2026-11-06. Initial brief/policy reads
  precede that measurement by a few seconds and are not charged separately.
- Both capture manifests are byte-identical. SHA256 of `live_before.sha256`:
  `1d0a6ffc50f93bbb47d281dce79e1f7bfdae9f0ba2c8319134de551478100cf6`.
  The queued CPU check verified all **45/45** snapshot files against those
  manifests. The manifest records original model paths; lines below are the
  same line numbers in `source/<filename>`.
- In this report, original model paths are relative to
  **`models/deepseek_v4_1_flash_a3_w8a8/`**. For example, `mla_proj.py:76`
  means that original model file and the frozen `source/mla_proj.py:76`.
  Every cited snapshot hash can also be resolved in
  [live_before.sha256](live_before.sha256).
- Authority: `goal.md` decisions **1–9**, with decision 9 overriding stale
  single-request/static-shape prose; checkpoint configuration and export basis;
  the recorded vllm-ascend A3 baseline. `INTERFACE.md` permits BF16 activation
  caches and replicated 32-head indexing. Those permitted differences do not
  authorize a new attention equation, moving Engram, reducing the 2048-block
  hierarchy, or static-only MoE. MODEL_SPEC/MODULES/WEIGHT_MANIFEST/INTERFACE/
  SERVING_ABI/ACCEPTANCE, HANDOFF, latest Observer directives, and the previous
  reference study were read. The A5 and DSpark study is context, not proof of
  A3 numerical equivalence or a runtime fault.

No local vllm-ascend checkout was found in the inspected workspace. An official
upstream archive was fetched **into this review directory**, after sourcing
`~/proxy.sh` for network commands:

| Reference | Revision and preserved evidence |
| --- | --- |
| Agreed vllm-ascend baseline | Tag **`v0.30.0rc2`**, resolved by `git ls-remote` to **`8fde6d4e335fe300e2b949aab0e31659cce0811d`**; [tag resolution](vllm_refs.txt), [extracted source](upstream_rc2), archive `vllm_ascend_rc2.tar.gz` SHA256 **`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`** |
| Official tutorial at that revision | [DeepSeek-V4.1-Flash](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md), preserved under `upstream_rc2/docs/source/tutorials/models/` |
| Attention equations and source positions | [dsa_v41.py](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/vllm_ascend/attention/dsa_v41.py), abbreviated **U-attn** below |
| Model order, topology and projections | [deepseek_v41/model.py](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/vllm_ascend/models/deepseek_v41/model.py), abbreviated **U-model** |
| Candidate hierarchy | [deepseek_v41/indexer.py](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/vllm_ascend/models/deepseek_v41/indexer.py), abbreviated **U-index** |
| Rotated-checkpoint Engram gate | [engram/common.py](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/vllm_ascend/models/deepseek_v41/engram/common.py), abbreviated **U-engram** |
| Current-version check | Main resolved to **`9649d5a76209159b773a46275c95ea743e44beb7`**. Its [attention source](https://github.com/vllm-project/vllm-ascend/blob/9649d5a76209159b773a46275c95ea743e44beb7/vllm_ascend/attention/dsa_v41.py) is preserved as `current_dsa_v41.py`. It adds replay-visible-length handling and KV-only/replay support relative to rc2. The checked equations remain: no post-Wq_b Q RMSNorm (`:402`), inverse output RoPE (`:817`), and C2 source position `input_positions-1` (`:1515`). Findings use rc2, not those later replay additions. |

The checkpoint's `config.json` and README were read without loading full
weights. The README's table-storage prose is stale per decision 7, but its
Engram basis contract agrees with both `engram_rotation_config` and U-engram.
[reference_inputs.sha256](reference_inputs.sha256) hashes inspected primary
inputs. The installed environment does not supply the original HF V4.1 forward;
attempts at the named Transformers 5.6.0 source path and the vLLM verified-commit
model path returned HTTP 404. Thus **no original HF execution or exact-checkpoint
generation baseline is claimed**. The official, pinned Ascend implementation is
available and directly supports the semantic findings below.

Initial network attempts included an SSL connection failure, a GitHub API rate
limit (HTTP 403), and incorrect raw-file paths (HTTP 404). They were not model
failures. Preserve `vllm_tag_response.txt`, the 404 response files and
`network_error.txt`; the successful git ref resolution/archive supersede them.

## Verified findings, severity ranked

**P1** means a supported input can change model results, invalidate state, or
prevent required generation/dynamic execution. **P2** means a narrower defect
or an already-repaired snapshot defect. These are source/CPU findings; none is
a newly observed NPU result.

### F01 — P1: MLA adds a Q normalization absent from the reference

- **Location:** `mla_proj.py:71–79`, `mla_proj`, especially
  `rms_norm_fp32_unit` at **`:76`**; golden `_golden_mla_proj` **`:173–174`**.
  Snapshot SHA256:
  **`7f4a6470ef40522b8a8e569bda403d874a4e432266fa6399c16567b78f29909e`**.
- **Trigger/behavior:** any Wq_b head whose RMS is not 1. The kernel changes
  `q = Wq_b(RMSNorm(Wq_a(x)))` to `q / RMS(q)` before RoPE. This discards learned
  per-head magnitude and changes attention logits relative to the sink as well
  as to other keys. It affects every attention family in all 40 layers.
- **Expected:** U-attn `_project_q` **`:320–324`** applies Wq_b then RoPE, with
  no head RMSNorm; its overlap path explicitly documents the same omission at
  `:444`.
- **Golden/coverage:** the actual candidate golden performs the same extra
  normalization, and mode/layer/full-forward goldens import it. A queued CPU
  counterexample runs that frozen golden: candidate Q is
  `[0.166992,0.5,0.992188,1.65625]`; the independently composed W8A8 projection
  without post-Wq_b normalization is `[0.734375,2.203125,4.375,7.3125]`.
- **Action:** remove the unintended normalization from the common compute and
  correct the semantic reference before reaccepting all consumers. Separately
  calibrate the Wq_b/RoPE cast boundary against the pinned W8A8 implementation.
  **Confidence: high.**

### F02 — P1: every attention chain omits inverse RoPE on the attended value

- **Location:** representative `prefill_swa.py`,
  `prefill_swa_attention` **`:104–110`** (SHA256
  **`9a372a5099499ed124eb1590719de31da2b6a7616b9fdf4964ac2444854ce9f3`**);
  `decode_c2a_full.py`, `decode_c2a_full_attention` **`:184`** (SHA256
  **`5ea2b9183532cef16e6d6ea0b0ea1d30a7b14ff917558fc7e71f5f91dbec16cf`**).
  The same direct `attention_core -> o_proj` sequence occurs in all distinct
  prefill/decode full, reindex, reuse and SWA chains. `decode_c1a_reuse` imports
  the shared reuse computation.
- **Trigger/behavior:** nonzero query position with a nonzero attended rotary
  tail. Raw KV is rotated in `mla_proj`; the weighted sum of those values is
  sent directly into Wo_a/Wo_b in the rotated basis.
- **Expected:** U-attn **`:711–716`** explicitly applies the **inverse** of the
  query-position rotary transform to `attention_output`, then projects it.
  This is a value/output transform and cannot be replaced by rotating Q/K.
- **Golden/coverage:** `prefill_swa.py:258–270`,
  `decode_c2a_full.py:667–673`, and `layer_reference.py:_partial_rows:292–308`
  all pass the raw attention result to `golden_o_proj`. The original Wo weights
  cannot absorb a position-dependent inverse rotation at load time.
- **Action:** restore the shared output transform in every actual chain and
  its independent golden; use nonidentity RoPE and asymmetric Wo weights in
  coverage. **Confidence: high.**

### F03 — P1: compressed attention KV is published without source-position RoPE

- **Location:** `prefill_c2a_full.py:128–138`, `prefill_c2a_full_attention`
  (SHA256 **`6c1114067306b859f9beefba08837e4df857b00382c8597528e2aec81b961429`**);
  `prefill_c1a_full.py:123–130`, `prefill_c1a_full_attention`
  (SHA256 **`59c423b97d100b68ca50f4601aef0c244fa233b96e48f17b46453e6ce426f6dc`**).
  Both decode full-source chains do the same.
- **Trigger/behavior:** a source emits a compressed row at a nonidentity
  rotary position, and any source/reuse/reindex layer selects it. The normalized
  **unrotated** latent is written to `compressed_cache`. Only the separate
  index key is rotated.
- **Expected:** U-attn **`:544–564`** first constructs index K from the
  unrotated latent, then rotates the latent's last 64 lanes before publishing
  long attention KV. Both consumers are necessary.
- **Golden/coverage:** `decode_c2a_full.py:627–628` directly scatters the
  unrotated compressor output; `layer_reference.py:382–384` repeats it.
  Thus current compressed-cache equality gates actively accept the wrong basis.
- **Action:** retain the unrotated latent for index-key projection, publish a
  separately rotated attention latent, and update all source/cache goldens.
  **Confidence: high.**

### F04 — P1: C2 rotary positions use compressed row numbers instead of original group origins

- **Location:** `generate.py:_compressed_rope_rows` **`:932–938`**,
  `prefill_fwd.py:1752`, `decode_fwd.py:1622`, and the corresponding compressed
  table construction. Generator SHA256:
  **`3d208c4390c490df2665206471c4e697b8a418cb3778ccd50e632ed5306ede33`**.
- **Trigger/behavior:** a completed pair after the first pair. At emitting
  token position **1253**, the generator chooses rotary row **626** because
  it uses `compressed_lens-1`.
- **Expected:** U-attn metadata **`:1340–1349`** uses original source position
  **`position-1`**, i.e. **1252**, for both compressed index K and long KV.
  C1 uses the current original token position (`:515–519`). A compressed cache
  row number is an address, not the rotary coordinate. The generator's own
  unused `compressed_key_tables:763–767` also records ratio multiplication.
- **Golden/coverage:** fixtures pass these wrong tables to both kernel and
  golden, so shared-table comparisons cannot detect the coordinate error.
  F03 must also be corrected; fixing the table alone still leaves attention KV
  unrotated.
- **Action:** derive C2 source RoPE from absolute group-origin positions,
  including odd-start continuation; test original coordinates independently of
  physical pages and compressed row IDs. **Confidence: high.**

### F05 — P1: Engram gates a rotated hidden stream against original-basis keys

- **Location:** `engram.py:71–82`, `engram_gate`, and
  `golden_engram:186–193`; SHA256
  **`c17a0bb88589a8fff13c35ffa4cb0b5546d8001cdf29eafed2e743e0ce3a861a`**.
  `weight_loader.py:303–305` only folds elementwise `q_weight*k_weight`.
- **Trigger/behavior:** real checkpoint Engram layers 1/14 with nonidentity
  global rotation and nonzero rows. The gate uses `h_rot * (q*k) * key_original`.
  No gate-input inverse rotation is accepted or applied anywhere in this path.
- **Expected:** checkpoint README and `config.json:engram_rotation_config`
  keep K/gate in the original basis and fold rotation into V only. U-engram
  **`:55–72`** restores hidden with `h_rot @ Q.T` for the gate, then updates
  the rotated residual with the already-rotated value. An elementwise q*k fold
  cannot absorb a general rotation into a diagonal channel weight.
- **Golden/coverage:** candidate golden repeats the wrong product; fixtures
  contain neither a rotation argument nor a rotated-basis reference. The queued
  CPU probe executes candidate and pinned upstream gates: candidate first stream
  `[1.765625,0.765625]`, original-basis gate `[1.50025,0.50025]`.
- **Action:** implement the checkpoint's gate-basis restoration (or a proven
  equivalent transform), with no second rotation of V. Test a nonidentity
  rotation. **Confidence: high.**

### F06 — P1: Engram is applied after attention instead of on the layer input

- **Location:** `prefill_fwd.py:_prefill_encoder_layer` **`:421–460`**,
  especially **`:453–457`**, SHA256
  **`7810211686472e65a1f89f524b64893b5228fa5753bba4c6013c6e5ef9ef358e`**;
  `decode_fwd.py:523–530`, SHA256
  **`f7f418d5cfa2f3f4496379a269e99d1f2596c86681c655769ef0be734685f4b2`**.
  Layer-suite prologs use the same order.
- **Trigger/behavior:** nonzero Engram delta at layers 1/14. Candidate runs
  attention/HC post first, then Engram, then FFN. Attention queries, source
  compressor state and caches therefore see a stream without that layer's
  Engram update. At layer 14 this also contaminates the shared C2 source.
- **Expected:** U-model **`:1565–1592`** applies `layer.engram` to the incoming
  HC stream **before** calling that layer; the layer then performs attention
  followed by FFN (`:888–919`).
- **Golden/coverage:** `layer_reference.py:404–422` applies Engram after its
  attention TP sum/HC post. Both forward goldens compose this same function.
- **Action:** move the common Engram boundary before attention/HC pre in
  actual forwards and layer validation, and independently verify the first
  affected cache boundary. **Confidence: high.**

### F07 — P1: the 2048-block candidate hierarchy is replaced by unused 64-block diagnostics

- **Location:** `config.py` **`:191`** (`CANDIDATE_POOL_BLOCKS=64`, SHA256
  **`89769114dfe8965bf133496d283244ea7a9d284c3186321daafc406c4731ddb0`**);
  `decode_c1a_full.py:55,154` (`2048//32`, SHA256
  **`0ee9dde2e0a0f7a7b87109d919cc604997a2e72bde0e0f03bb07dd3e81a0d4fa`**);
  `decode_c1a_reindex.py` **`:114–130`**, SHA256
  **`5273f838d8bf9eeb6f0cd2c827a8698df26baaa4c4e339d13560fe7035539b16`**.
  Prefill has the same hierarchy omission.
- **Trigger/behavior:** long history with more than 64 candidate blocks, or
  a reindex query whose strongest row lies outside layer 20's source pool.
  Layer 20 caps output at 64 blocks. Layers 24/28/32/36 rescore **all** history
  and compute a new pool rather than restricting Top-K to the source candidates.
  `decode_fwd.py:745` supplies layer 20's pool output, but `:773` supplies a
  different output slot to each reindex; no source-pool input is consumed.
- **Expected:** checkpoint `candidate_topk_blocks=2048`, block size 8,
  candidate source 20. U-model `build_layer_plan:538–549` marks later index
  sources as candidate-filter consumers. U-index **`:274–288`** selects mode 1
  at the source and mode 2 with `candidate_topk_index=candidates` at consumers.
  This is a model selection rule, not an optional latency optimization.
- **Golden/coverage:** `golden_index_select:440–470` computes global row Top-K
  before an independent pool, and reindex goldens call it. All-context Top-K
  therefore passes the wrong reference. With <=512 compressed rows the 64-block
  cap keeps all blocks, hiding the hierarchy distinction; bounded 64-cap
  acceptance cannot establish checkpoint 2048-cap semantics.
- **Action:** preserve the original source pool across reindex layers, enforce
  its counts/eligibility in shared selection, and restore the configured cap
  in real model paths. Add a forbidden-high-score-row negative control and a
  >64-block source case. **Confidence: high.**
- **Concurrent status:** the latest Observer directive assigns this repair to
  the main agent. This finding describes only the frozen snapshot; the live
  indexer already contains new repair work. No acceptance of it is implied.

### F08 — P1: tied MoE maxima create invalid expert IDs and remove several routes at once

- **Location:** `moe_gate.py:72–84`, `moe_gate`, especially **`:77–78`**;
  SHA256 **`c559151b8c29df5216f9118240e3409ab95558969680d3b4368dec59f7fa12e2`**.
- **Trigger/behavior:** equal biased scores, e.g. zero token input and equal
  gate biases. `is_max` contains multiple ones; `selected_index` is the
  **sum of all tied expert indices**, and `selected_value` sums their scores.
  All tied experts are then suppressed in one slot. With 384 equal experts the
  first emitted ID is **73536**, outside `[0,384)`. `moe_exchange.py:147` uses
  it as an array index; expert ownership/route capacity assumptions break.
- **Expected:** Top-6 must return six distinct valid expert IDs, with normalized
  weights from the selected original scores. Any defensible tie ordering is
  permissible; summed IDs are not. Checkpoint Top-K/noaux configuration and
  U-model `:236–252` establish the route contract.
- **Golden/coverage:** `golden_moe_gate:123` correctly uses `torch.topk`, but
  default random inputs/biases avoid ties. The CPU probe verifies the source
  mask-sum value 73536 versus six valid golden IDs. This is not merely a legal
  Top-K ordering mismatch.
- **Action:** choose and suppress exactly one valid index per slot; add tied
  and zero-activation cases and require unique IDs in `[0,384)`.
  **Confidence: high.**

### F09 — P1: complete forwards remain static in MoE and process inactive rows

- **Location:** `moe_exchange.py:101–106`, `moe_ep16`; `moe_ffn`
  **`:425–432`**, SHA256
  **`36016091d3727415534bae01b2f1db239c760cdbc20c2ba520e08fa274dffb6b`**;
  `prefill_fwd.py:1217,1240,1490–1494` (SHA above);
  `decode_fwd.py:175–178,540,554,840` (SHA above).
- **Trigger/behavior:** replay a compiled capacity T=16 with one active token,
  change actual T, or give different active counts to DP groups. Attention
  publication partly uses `token_counts`, but MoE is called with
  `capacity=rows, rows=rows` where rows is a trace-time constant. It routes and
  evaluates the whole bound. Embedding, norms, gate, HC and portions of state
  processing use tensor extent rather than the active count. Prefill explicitly
  caches a separate entry object per token bound. Decode mutates `_MOE_ROWS`
  before compilation; assigning it after compilation cannot change an artifact.
- **Expected:** decision 9 requires one artifact per phase with runtime token
  counts/context lengths, genuine multi-request batching and inactive-state
  isolation. `INTERFACE.md:7` also requires runtime MoE rows. Static capacity
  is legal, static active computation/state advancement is not.
- **Golden/coverage:** goldens iterate fixture tensor shapes and equal counts;
  `build_fwd_specs` sets counts to the full shape. No accepted same-artifact
  varying-T/count proof follows from DYN annotations or structural tile checks.
  Distinct DP inputs with equal counts do not cover idle/unequal DP lifecycle.
- **Action:** carry per-rank active counts through every stateful/collective
  boundary, keep inactive rows untouched, and demonstrate same-artifact replay
  at different counts including an idle DP group. **Confidence: high.**

### F10 — P1, conditional: dynamic EP auxiliary writes overlap neighboring source lanes

- **Location:** `moe_exchange.py:176,178,195–198,229`, `moe_ep16`;
  SHA256 **`36016091d3727415534bae01b2f1db239c760cdbc20c2ba520e08fa274dffb6b`**.
- **Trigger/behavior:** use the primitive's runtime-row branch with
  `0 < tokens < capacity`. Destination lane starts are separated by **tokens**,
  but each `recv_aux` put writes **capacity** rows, including zeros. Adjacent
  `(expert,source)` auxiliary regions overlap. With tokens=1/capacity=16, lanes
  starting at 0 and 1 overlap on rows **1–15**. A later zero-padding write can
  erase another source's route/weight. Different source/destination token counts
  additionally disagree on `*tokens` offsets at push and compact.
- **Expected:** disjoint per-source reserved lanes, consistently addressed by
  both peers, sized for the stated worst case. Runtime counts must control
  payload lengths rather than changing a shared layout independently per rank.
- **Golden/coverage:** `golden_moe_ep16` uses Python route lists and cannot
  reproduce overlapping remote memory writes. Current full forwards force
  `tokens==capacity`, hiding this bug while violating F09. The CPU check proves
  the overlap geometry; **no particular device overwrite schedule is claimed**.
- **Action:** use a consistent capacity stride (and matching compact reads)
  or nonoverlapping actual-count transfers with exchanged peer metadata; cover
  capacity>T and unequal active DP counts. This is independent of the known
  decode compressor/pool publication symptom. **Confidence: high for unsafe
  geometry; device manifestation is schedule dependent.**

### F11 — P1: generation cannot bind the new per-profile RoPE ABI, and computes the wrong tables

- **Location:** `generate.py:707–727,804–815,990–1024,1027–1056`, plus
  `ordered_dispatch_args:494–505`; SHA256
  **`3d208c4390c490df2665206471c4e697b8a418cb3778ccd50e632ed5306ede33`**.
- **Trigger/behavior:** real generation with the delivered entries, which
  require `rope_cos_swa/rope_sin_swa`. Both input builders construct only
  `rope_cos/rope_sin`; neither plan nor weight assembly adds the SWA arguments.
  `ordered_dispatch_args` will reject a missing required value. Moreover,
  `resolve_rope_profile` returns **`per-profile`**, but `_rope_args_pair` only
  handles `per-layer` specially. `layer_rope_tables("per-profile", ...)` tests
  equality with `"compressed"` and therefore generates **base** tables for the
  main compressed-layer argument.
- **Expected:** bind both required profiles explicitly, assigning base to
  layers 0/1 and compressed to 2–39. U-model **`:611–624,700–705`** also disables
  YaRN for SWA. The generator instead applies YaRN globally (`:730–747`) based
  on an inference from config presence; that inference contradicts the pinned
  forward semantics. Config metadata alone does not prove where scaling applies.
- **Golden/coverage:** dry-run scans **provided** `prefill_values.items()`
  (`:1765`), rather than requiring every entry parameter. `mismatched_layers=[]`
  only reports the signature marker. The CPU probe executes the real resolver
  and constructor: main argument equals the base profile; required-name binding
  raises `missing dispatch value for 'rope_cos_swa'`. Its SWA YaRN probe at
  position 4096 also differs from the independently computed unscaled base.
- **Action:** validate complete ordered argument binding before expensive
  weight allocation/dispatch; build both profiles from the pinned equations,
  including the SWA no-YaRN rule. **Confidence: high.**

### F12 — P1: generation advances only position IDs and feeds a token tile as one decode token

- **Location:** `generate.py:1027–1056,2069–2107,2119–2121`, `device_run`;
  SHA256 **`3d208c4390c490df2665206471c4e697b8a418cb3778ccd50e632ed5306ede33`**.
- **Trigger/behavior:** consecutive greedy decode, after resolving earlier
  dispatch blockers. Metadata and RoPE are built once. The loop changes only
  `input_ids`, `position_ids` and a Python `_MOE_ROWS` variable. It leaves window
  slots/indices, compressed slots/lens/tables, both RoPE profiles and Engram
  lookups frozen. Later tokens overwrite old slots and attend using old masks
  and visible lengths; C2 emission parity becomes inconsistent.
- **Additional first-step defect:** `build_entry_contracts(args.tokens)` also
  gives decode the prefill tile count. `build_decode_state` plans **tokens**
  consecutive positions of one request and sets every `token_count=tokens`.
  The loop fills **all** IDs with the same generated token and **all**
  positions with the same scalar, then samples the last row. For T>1 this is
  neither one-token decode nor a valid multi-token continuation.
- **Expected:** each step consumes exactly the current active token per request
  and advances all position-dependent metadata consistently while preserving
  persistent cache contents. Decision 9 and `INTERFACE.md` spell out the slot,
  lens and RoPE contracts. U-attn consumes these as runtime inputs; it does not
  rebuild them from `position_ids` internally.
- **Golden/coverage:** full-forward fixtures run isolated plans, not this host
  loop. Handoff shape checks and state checksums cannot prove correct addresses.
  CPU execution of the actual metadata helper demonstrates slot **127 -> 128**,
  changed window indices, and C2 slot **63 -> -1** at that step; the loop updates
  none of them.
- **Action:** implement a per-step request/allocator/metadata update preserving
  caches; use an actual one-active-token decode plan for one request. Verify
  prefill -> at least three decode steps crossing C2 parity and a page boundary.
  **Confidence: high.**

### F13 — P1: the non-diagnostic Engram generation path discards future hashes and has no autoregressive hash update

- **Location:** `generate.py:1218–1253`, `load_engram_hash_ids` /
  `build_engram_lookup`; `device_run:1986–1988,2090–2091`; same generator hash.
- **Trigger/behavior:** `--engram-mode host-ids` with a prompt fitted to T
  tokens and decode requested. `load_engram_hash_ids(..., rows=args.tokens)`
  **truncates** every supplied tensor to `[:T]`, even if more rows were supplied.
  First decode requests `[position:position+tokens]` with position >=T, yielding
  an empty tensor; `ids.max()` raises. The CPU probe supplies 20 rows, retains
  16, and gets zero rows at the first decode position 16.
- **Expected:** Engram n-gram hashes follow each request's actual token history,
  including newly generated tokens. U-model `prepare_engram:1116–1178` and its
  hash-state helpers retain request history and exclude inactive tokens. A
  precomputed prompt-only table cannot supply arbitrary future generated hashes.
  The default zero-diagnostic path explicitly omits Engram and cannot satisfy
  decision 7's real-model scope.
- **Golden/coverage:** device component goldens start from already gathered
  rows. They cover table dequant/projection, not token-history hashing or the
  first decode lookup. Dry-run builds only prompt rows.
- **Action:** provide and validate the original hash/token-compression/history
  pipeline for prefill and each decode step. At minimum, stop truncating required
  rows; fixing truncation alone does not establish correct autoregressive hashes.
  **Confidence: high.**

### F14 — P2, conditional: routed-expert loader ignores its offset-fold result

- **Location:** `weight_loader.py:347–355`, `WeightLoader.expert_tensor`,
  specifically **`:354`**; SHA256
  **`437b2b2308aba48912592a0ae676cf12e8cee0a48b7ae20e81372cece6d77f54`**.
- **Trigger/behavior:** a representable nonzero routed-expert weight offset.
  `_fold_offset(weight, ...)` returns a new adjusted tensor at `:430–439`, but
  this caller discards it and NZ-packs the original payload. Dense `_w8a8`
  callers correctly assign the return. The CPU probe executes the actual
  method with a nonzero fold and observes unchanged bytes.
- **Expected:** under the loader's declared contract, the packed payload must
  represent `(q-offset)`, or a separate GEMM correction must be carried through.
  No such device correction exists. WEIGHT_MANIFEST explicitly claims offsets
  are folded.
- **Golden/coverage:** current weight/quant fixtures use symmetric zero offsets;
  the recorded 201-tensor checkpoint sample found zeros. Consequently this is a
  verified conditional loader bug, **not proof of corrupted weights in this
  particular checkpoint**. No exhaustive offset read was done in this review.
- **Action:** retain the returned payload and independently test nonzero
  representable offsets; preserve explicit failure for unrepresentable folds.
  **Confidence: high; checkpoint incidence unestablished.**

### F15 — P2, already repaired live: generic indexer pool width truncates pages to eight rows

- **Location:** frozen `indexer.py:352–356`, `lightning_indexer`, especially
  **`:354`**; SHA256
  **`7cdb9b5b421dac20b46679f1b917055d01e8c38d8201dff66158546d4d22f1c7`**.
- **Trigger/behavior:** compressed length exceeds `table_width*8`. Candidate
  pool selection receives width `table_width*BLOCK` (candidate block 8), while
  the score tensor and physical cache use page size **128**. Pool IDs/counts
  truncate independently of correct score/row Top-K output. Invisible trailing
  score pages also remain unwritten in the snapshot (`index_scores:141–160`),
  whereas its golden initializes all scores to zero.
- **Expected/golden:** `INTERFACE.md`, paged address math and the independently
  pinned block/cache distinction require page-derived score width. The frozen
  golden uses the full score width and can expose the mismatch.
- **Live status:** read only to label repair status: live `indexer.py` at epoch
  **1793930809** has SHA256
  **`b8f61cbf949083a43ef9c566aca402f6235f65164076f09a85c127338012b46a`**,
  with `BLOCK_SIZE` and trailing zero stores already present, plus ongoing
  reindex work. The Observer separately records acceptance of the narrower
  completed repair at hash `67cf15fc...`. This review neither reruns nor accepts
  that live work. Existing phase chains already passed page width explicitly;
  this generic-wrapper defect alone does **not** explain their old composed
  decode publication failure. **Confidence: high.**

## Golden audit and missing evidence

These gaps are distinct from the verified candidate equations above:

1. **The golden dependency graph is not independent model semantics.** Mode
   goldens import `_golden_mla_proj`, compressor and index selection goldens;
   layer goldens compose the same functions; complete-entry comparators reenter
   those layer goldens. F01–F07 are therefore reproduced at every level. Reusing
   a helper is good engineering only after its rule is independently grounded.
   A full-model PASS against this graph cannot establish original-model parity.

2. **EP ownership is not challenged by distinguishable owner weights.**
   `moe_exchange.py:438–488`, `golden_moe_ep16`, groups routes by `(dst,local_e)`
   but selects weights by **`local_e` only**, ignoring dst. Layer helper
   `layer_reference.py:_moe_weight_shard:130–140` strips the rank axis by taking
   rank 0. That is valid only for the explicitly uniform fixture, not actual
   EP16 ownership of experts `[24*r,24*r+24)`. Full fixtures fill expert tensors
   with common constants (`decode_fwd.py:1571–1585`,
   `prefill_fwd.py:1696–1708`). Distinct source DP inputs do not distinguish a
   wrong owner if every owner's corresponding expert is identical. A future
   acceptance fixture must differentiate owner, local expert and layer, keep
   distinct DP requests, and verify every route is returned once.

3. **TP/vocabulary coverage remains insufficient.** The source rank mapping,
   16-head/2-output-group slicing, 32320-row vocab partition, and `group_base`
   addressing are consistent on inspection. But prefill fixture IDs are drawn
   only from `[0,LOCAL_VOCAB)` (`prefill_fwd.py:1769`); its composed golden indexes
   that rank's table directly (`:2081–2085`). Decode's golden assumes rank-
   repeated embedding weights and uses `global_id % LOCAL_VOCAB`
   (`decode_fwd.py:1931–1938`), rather than fetching the owning TP rank's actual
   table. Neither is an independent arbitrary-shard embedding golden. All TP
   vocabulary intervals and distinct head/group partials need owner-aware
   acceptance. `tp_reduce_test` is still OPEN, not cleared by this review.

4. **Staged comparators are diagnostic, not end-to-end independence.**
   `prefill_fwd.py:2093–2095,2159–2187` and decode counterparts replace each
   layer's input with preceding device snapshots before computing staged
   expectations, including the logits expectation. This locates local errors
   but can condition later references on earlier wrong state. Keep such gates
   and add an independent bounded multi-layer trajectory/handoff check. No
   threshold change is suggested.

5. **Current fixtures do not demonstrate decision-9 lifecycle limits.**
   Complete prefill is one request at start 1248; default decode is two one-token
   requests at 1250/1253. These are useful nonempty-history cases, but do not
   prove cold prefill, same-artifact varying T/B/S, unequal DP/EOS/idle counts,
   long candidate restriction, permuted physical page allocation, or three-step
   prefill/decode continuation. `_rank_repeat` replicates much metadata/state;
   independently varied DP state is required to expose cross-group leakage.

6. **Real checkpoint execution remains unproved.** This review reads metadata
   and source, not full tensors. It did not validate all loaded offsets, all
   layer expert bindings, peak loading copies, or full generation. The driver
   eagerly stacks all per-layer/per-rank experts on the host before the resident
   upload (`generate.py:599–610,634–650,1319`), so the decided resident ABI alone
   is not evidence of bounded peak host memory. Original HF generation fixtures
   remain uncalibrated per ACCEPTANCE. Diagnostic zero-Engram output cannot
   close real-model acceptance.

## Suspected issues and questions requiring targeted evidence

- **W8A8 route-weight/cast boundary:** `moe_experts.py:63–72,126–133` multiplies
  the route weight into SwiGLU before BF16 rounding, dynamic INT8 quantization
  and W2. In real arithmetic it commutes with W2; in quantized arithmetic the
  rounding boundary can differ from output-weighted combine. The golden mirrors
  it. The pinned U-model specifies normalized routing and routed-output scaling
  (`:247–250`), but the actual fused W8A8 path must be traced/calibrated before
  declaring this an unacceptable numerical defect. Do not silently change it
  based on an unquantized algebra argument.
- **Low-amplitude activation quantization:** `INT8_AMAX_EPS=1e-4` in config is a
  real quantization floor shared by kernel/goldens. The upstream path delegates
  to `torch_npu.npu_dynamic_quant`. No NPU comparison for rows below that floor,
  exact half ties, or all-zero rows was performed. INT32 GEMM accumulation,
  row/column scale multiplication and the host NZ pack/unpack are internally
  consistent on inspection; CPU inverse roundtrip does not prove hardware NZ
  or rounding conformance.
- **Unwritten C2 output rows:** `compressor.py:100,110–136,161` only assigns
  `latent` for odd positions, then normalizes every tensor row. The golden
  explicitly initializes even rows to zero. Negative compressed slots usually
  prevent these rows reaching persistent caches, so this is not asserted as the
  known missing-state root cause. Confirm allocation/valid-region semantics or
  explicitly define nonemitting output rows before treating full-output equality
  as robust evidence.
- **State-ring equivalence:** baseline uses a 32-row FP32 state ring; agreed
  local ABI uses two rows and writes only the request tail. For strictly
  increasing consecutive ratio-2 continuation, only the previous even token is
  required and the narrower state can be sufficient. Rewind/replay/speculation
  equivalence is not proven; DSpark/MTP exclusion does not itself prove all
  serving replay requirements. The current odd/even decode publication symptom
  remains a separate empirical failure.
- **Runtime attribution remains unresolved:** no writer-occurrence, actual
  runtime scalar, pointer/offset/owner/version or retirement evidence was
  collected. The study's generated stores establish code presence, not their
  execution. None of F01–F15 proves the owner of the old composed state/pool
  failures. Preserve those gates and the existing negative-probe matrix.

One rejected hypothesis: a source read initially suggested INT32 overflow in
the embedding `gap*gap` owner calculation. A targeted read of the historical
generated `embed_shard_ids.cpp:72–93` shows INDEX/int64 arithmetic, so this is
**not** reported as a defect. Its file hash is
`d5f6556d413fdbc31cd28fea9e082274f4d6e936c3a3fc2d40f715ada0d81c21`
under the already-recorded `081644` decode build. This is historical lowering
evidence, not a claim that the build is the frozen snapshot.

## Coverage matrix

| Area | Reviewed evidence | Result / practical limit |
| --- | --- | --- |
| Configuration, source/reuse schedule | checkpoint config; config.py; full encoder/decoder source slot bindings; U-model layer plan | 40-layer modes and source identities align; candidate filtering does not (F07) |
| W8A8 scale/offset/layout | quant.py, gemm_w8a8.py, utils.py, weight_loader/map, expert projections; pinned W8A8 method | INT32 accumulation and per-token/per-channel scale structure align; conditional offset bug F14; hardware quant/cast/NZ parity not newly tested |
| MLA and RoPE | common projection, rope.py, all distinct mode chain endings, source publication and goldens; U-attn | F01–F04, F11; interleaved adjacent-pair rotation itself is consistent |
| Attention softmax/sink | attention.py kernel and golden; U-attn scale/sink interfaces | online denominator includes sink and visible-key masks; output basis is wrong (F02); no new device precision verdict |
| Compression/state | compressor.py kernel/golden; source chains; pinned compressor/ring reference | pair pooling and request-tail ownership support ordinary consecutive requests; nonemitting rows and replay equivalence remain questions |
| Index scores/top-K/candidates | indexer.py, C1/C2 full/reindex/reuse and goldens; U-index | F07, F15; positive common score scaling omitted versus upstream is ranking-invariant in exact arithmetic, so not separately labeled a selection bug |
| Paged masks/slots/DP isolation | utils plan/addressing; paged_cache.py gathers/publication; forward rank slicing; generator | physical row mapping/negative masking coherent for valid plans; F09–F12 and missing unequal/idle-DP replay evidence |
| HC/norm | rmsnorm.py, hc.py kernel/golden, layer sequencing; upstream HC reference helper | delayed pre-mix and Sinkhorn coefficient equations agree in a bounded CPU cross-check; Engram ordering violates input stream semantics (F06) |
| Engram | table dequant/accessor; projection/gate; loader q*k; generation host lookup; checkpoint rotation metadata; U-engram/U-model | F05, F06, F13; group-32 FP32 scale -> BF16 table dequant formula aligns with decision 7 |
| Routing/expert/dispatch/combine | gate and expert kernels/goldens, shared EP exchange, wrappers, full-forward call sites; upstream router/model | F08–F10; source capacity holds under unique top-k and equal bound, but owner-sensitive and lifecycle coverage are missing |
| Embedding/head | embedding.py, full-entry owner mask/reduce/seed, lm_head.py, loader vocab/head/group slices, host argmax | source partition dimensions/group mapping coherent; TP owning-shard goldens/coverage inadequate; no new device TP proof |
| Prefill/decode state progression | both full forwards, per-source selection reuse, generator handoff/loop | snapshot contains compiled-stack composition; real dynamic execution and generation state progression fail F09/F11–F13 |
| Goldens/acceptance provenance | mode/layer/full/staged reference graph; fixture construction; historical report and current directives | shared wrong semantics demonstrably pass candidate-oriented goldens; no original HF execution or new NPU acceptance |

## Reproduction and closeout

The bounded [host_checks.py](host_checks.py) uses AST extraction to execute
frozen Torch golden functions without importing model/PyPTO modules. Reduced
dimensions are algebraic counterexamples, **not** substitute acceptance shapes.
It reads checkpoint config only, runs no model weights and never accesses NPU.
The actual pinned upstream CPU Engram and HC functions supply independent
comparisons. The queued final run verified:

- 45 snapshot hashes and identical capture manifests;
- candidate-golden extra Q normalization and missing Engram basis restoration;
- invalid tie-derived expert index, overlapping EP auxiliary lane geometry;
- actual host metadata changes at 127 -> 128;
- ignored expert offset-fold return and first-decode empty Engram hashes;
- wrong per-profile fallthrough, missing required SWA binding, unwanted SWA
  YaRN, and wrong C2 source coordinate;
- agreement of the HC coefficient equation with the upstream test reference.

Exact final command (from repository root):

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/logic_review_1793929657/host_checks.py'
```

Final task **`task_20261106_101043_119692330493`**, **exit 0**;
[complete queue/output log](host_check_queue_complete.log),
[structured results](host_results.json). Earlier successful bounded runs are
preserved in `host_check_queue.log` and `host_check_queue_final.log`; the script
was extended when further source concerns were found. No failed host assertion
was suppressed and no threshold was changed. The checks intentionally assert
that counterexamples differ; exit 0 means the defects were reproduced, not that
the candidate model passed.

**Resumption state: PARKED after completed independent review.** The observer
alone may relay findings into TASKS and assign fixes. Recommended repair order:
independently ground the common attention equations/goldens (F01–F04), Engram
basis/order (F05–F06), then hierarchy/routing and real dynamic lifecycle
(F07–F10); repair and host-validate generation binding/state/hashing
(F11–F13) before any expensive generation. The live generic-indexer repair is
already separately tracked; do not overwrite the concurrent main agent's work.
All historical open acceptance gates remain OPEN. This reviewer owns no active
device allocation, queued/running CPU task, or background process. End epoch
and final artifact hashes are recorded in `closeout.txt` and TIME_LOG.
