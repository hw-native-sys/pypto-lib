# Repetition model-logic review

Review-agent; bounded source/log/JSON investigation. Decision: **QUALITY NOT PASSED; root cause not established.** One concrete original-spec/kernel/golden discrepancy is evidenced and is a justified next numerical investigation. No implementation changes or new device runs.

## R01 — Original BF16 HC residual rounding is absent in kernel and golden

**Finding (high source confidence):** the implementation preserves FP32 HC residuals between sublayers, while the pinned original rounds each HC-post result to its BF16 sublayer dtype. This occurs after both attention and FFN, across 40 layers. Changing storage to FP32 is not inherently wrong if the required BF16 boundary is emulated; this implementation does not emulate that boundary.

Current source locations (all under `models/deepseek_v4_1_flash_a3_w8a8/`):

- `hc.py:192-215`, `hc_post`: casts BF16 sublayer input to FP32, mixes FP32 residual streams, then `pl.store(value, ...)` without BF16 cast. `hc_post_test:268-272` explicitly declares FP32 residual/output.
- `decode_fwd.py:535` creates FP32 `x_next`, `:538` invokes the shared `moe_ffn`; `moe_exchange.py:392` declares FP32 `x_hc_out` and `:434` passes it to `hc_post`. Attention/FFN layer interfaces also use FP32 HC intermediates (`prefill_layer.py:143`, `:1049`). Thus this is in the real composed forward, not only standalone tests.
- `hc.py:328-331`, `golden_hc_post`, stores FP32 `update + skip`; `layer_reference.py:152-155`, `_hc_post_reference`, returns FP32 `update + skip`. The composed layer uses it after attention (`:431`) and FFN (`:463`). Existing kernel/golden agreement can therefore accept the same omitted rounding boundary.

Original source paths below are relative to `logs/logic_review_1793929657/upstream_rc2/`:

- `vllm_ascend/models/deepseek_v41/model.py:1551-1553` seeds HC by repeating embedding rows; checkpoint config specifies BF16. `:886-921` sends attention and FFN sublayer outputs through `self.hc_post` and carries those residuals. `:871-877` calls the actual `npu_hc_post` operator.
- `csrc/moe/hc_post/op_host/hc_post_proto.cpp:52-56` sets output dtype to sublayer input dtype (`xDtype`).
- **A3 implementation** `csrc/moe/hc_post/op_kernel/hc_post.cpp:52-57` selects `HcPostKernelDSplit<DTYPE_X, DTYPE_POST>` outside DAV_C310. `op_kernel/hc_post_d_split.h:345-350` casts FP32 accumulation into T1 using `CAST_RINT` for the BF16 input case.
- Independent original reference `tests/deepseek_v41_utils.py:266-270` returns `y.to(x.dtype)`; original unit test `tests/ut/models/test_deepseek_v41_hyper_connection.py:199-210` explicitly includes this cast in its expected equation.

`evidence.json` verifies the original model, HC-post proto/A3 kernel, reference utility and RoPE source equal the stored rc2 archive members byte-for-byte. Member prefix is `vllm-ascend-8fde6d4e335fe300e2b949aab0e31659cce0811d`; archive SHA256 matches the prior pinned review (`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`). This extracted archive is not its own git checkout; a `git -C` lookup would walk to the parent repository and must not be used as its revision evidence.

**Model implication:** the omission changes the input to the next coefficient projection, collapse, RMSNorm and subsequent W8A8 quantization. It can affect routing, attention/cache values and final logits. Extra precision does not establish original-model equivalence. The log cannot determine whether this discrepancy changes a greedy decision or causes the repeated phrase. No corrected model execution, real activation differential, or original-model parity is claimed.

## Tiny CPU proof of the shared-reference discrepancy

Owner-requested task `task_20261108_004524_320911724075` ran through `task-submit --no-device --max-time 120 --run 'bash experiments/dsv41-flash-w8a8-a3/logs/repetition_logic_review_1794069747/run_hc_boundary.sh'`, using `activate-pypto.sh`, Torch `2.14.1+cpu`, and completed with **exit 0** at epoch 1794069930.5755281. `hc_boundary_probe.py` executes the unchanged actual original `hc_post_reference` and actual current `golden_hc_post` function ASTs, avoiding heavy module imports; this is not an invented replacement reference. Function-source SHA256 values are retained in `hc_boundary_result.json`.

One token, four HC streams, hidden width 5120. Inputs are BF16-exact signed units. Original residual dtype is BF16; current residual dtype is FP32 with identical BF16-exact values, preserving both functions' contracts. Post/residual coefficients are FP32; residual mixing is identity. For post `[1/512,1/256,3/512,1/128]`, the original output magnitudes are `[1,1,1.0078125,1.0078125]`, while the current golden returns `[1.001953125,1.00390625,1.005859375,1.0078125]`. **15360/20480 values differ; max absolute difference is 0.00390625.** Rounding the current output to BF16 exactly reproduces the original result. This quantifies the omitted boundary on synthetic data; it does not measure real-model divergence or execute the kernel on NPU. Exact kernel store `hc.py:214` and the FP32 output contract independently establish why the kernel also omits that boundary.

No individually justified/approved FP32 HC-residual exception was found in `goal.md`, `MODEL_SPEC.md` or `MODULES.md`. Goal decision9 and `MODEL_SPEC.md:185-187` permit A3 quantization/topology divergences and require other exceptions to be individually justified in MODULES. MODULES lists HC, padded coefficient rows and tests, but records no residual precision exception. This finding is bounded to those agreed policy records; an unlocated approval is not presumed.

## What the actual failed stream establishes

Source scope is source45CC `cc15877f8f4b0104a842f915d2434d9a2c9cd6f6752d6364d5d7bada26e21d8c`; `generate.py` SHA256 independently matches `41651e5af3b94223b645d4bfb52010e52ea7350aa3f3637cbcca6e29e093e609`. The failed case is `native_longer_real_generation_1794064297`, not an F09 draft or new run. `generation.json` records:

- Prompt `[16600,4465,260]`; first token 1014 at logit 26.6715488; 15 decode calls at positions 3 through 17 with previous sampled token as next input.
- Output IDs `[1014,1613,21908,515,260,1722,260,1722,260,1722,260,1722,260,1722,260,1722]`, text ` time...there was a way a way a way a way a way a way`.
- Every step changes the window cache, C1 compressed/index cache, C2 state cache, final `x_hc` and `pre_mix` hashes. C2 compressed/index hashes change at odd source positions and remain unchanged at following even positions, consistent with ratio-2 publication.
- The repeated 260→1722 transition has differing logits: p7=23.7532749, p9=25.9015331, p11=29.2979107, p13=28.0479488, p15=27.8781109, p17=28.1797676. These are evolving computed outputs, not an identical stale sampled record. Four DP votes agree, as the independent failed-case audit also reports.

These observations weaken a frozen whole-state/output explanation. They do **not** prove correct per-layer cache contents, cache addressing, causality, numerical parity, or absence of a common DP error. Hashes are truncated SHA1 over FP32-promoted full tensors (`generate.py:223-225,:1832-1844`); no per-slot tensors are retained. The per-shard maxima do not include the runner-up inside the winning shard, so even the greedy logit margin cannot be reconstructed.

Host continuation source also updates metadata/RoPE without clearing persistent caches (`generate.py:1103-1112`), and seeds new token HC state from embedding on every forward (`decode_fwd.py:874-923`). Cross-token persistence of final HC streams is not intended. The known active-less-than-capacity F09 trigger is absent from this all-active N3 / decode1 stream; it cannot be assigned as this case's root cause.

## Narrowed checks and next decisive evidence

Inspection found the local base/YaRN frequency equations (`generate.py:751-771`) consistent with pinned `rope_dsv4.py:334-384`; explicit base/compressed profile arguments exist. This is source agreement, not actual runtime-table verification. The apparent FP32 router concern is not another demonstrated deviation: original `rms_norm_cast` preserves the BF16-rounded result promoted to FP32 (original unit test `test_v41_rms_norm_cast_preserves_rounded_routing_input`), and the local router consumes that rounded representation. No new finding about loader offsets/NZ or expert weight values is established without payload evidence.

**Minimal proposed next investigation, requiring later owner authorization:** use the exact failed-case prompt and teacher-force its existing token prefix, with Engram explicitly disabled on both original and local paths. First compare one full prefill+first decode trajectory against the pinned original, retaining per-layer post-attention and post-FFN HC outputs for only the last prompt row and first decode row, plus head logits/top-two within every TP shard. Test the HC boundary directly by comparing each original post result with BF16-rounded local FP32 output promoted back to FP32, and quantify the first divergence. If those snapshots validate R01 as the first meaningful divergence, evaluate the original-boundary candidate across the same fixed prefix, including the first repetition transition p8→p9. An early p3 capture alone cannot prove the later repeated suffix is repaired.

Capture the live cache rows read/written at that first divergence (source layer, physical slot, previous-value/current-value, position/RoPE profile), rather than relying on another whole-cache hash. Compare the first differing logits and true global top-two margin to establish greedy impact. This is a targeted original-spec differential, not a newly invented implementation-matching reference, tolerance relaxation, prompt change or sampling workaround. No such real-model capture or run was performed here; the synthetic CPU HC microproof above is the only new numerical test.

The monitor PermissionError/queue exit is a separate outer failure; this review does not reattribute its unknown PID/path/epoch. Prior Hello/world EOS success supplies no missing original-model parity. Engram-disabled scope, plain short prompt and greedy decoding neither prove nor explain the degeneration.

## Review boundary and handoff

Read shared instructions, model rules, problem-handling rules, model documentation, agreed goal, current source, pinned archive/source, actual result metadata and a bounded progress-log inspection. No checkpoint tensor payload, NPU allocation, full-model golden, source edit, shared documentation/ledger edit, issue, commit or external communication. All writes are confined to this new review directory. No newly suspected upstream compiler/runtime fault is asserted; R01 is an own-model source/golden semantic discrepancy.

Artifacts: `REPORT.md`, `evidence.json`, `source_manifest.json`, `hc_boundary_probe.py`, `run_hc_boundary.sh`, `hc_boundary_result.json`, `task_submit.log`, `work_record.json`, `artifact_manifest.json`, `PARK.json`. Root owns any one-time import into shared records and any later change/test decision. **PARKED at source-evidence milestone; no process or device task remains.**
