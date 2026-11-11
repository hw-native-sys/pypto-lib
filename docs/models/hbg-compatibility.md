# HBG model compatibility

This page tracks the HBG model paths on the Lib `dev/hbg` branch. A
historical four-repository result and validation of the current branch are
recorded separately. Graph means explicit recording and replay. Async
qualification covers request state, completion, and retirement in Serving.

| Platform and model | Historical validation | Current branch coverage |
| --- | --- | --- |
| A3 Qwen3-14B BF16 | The integrated 40-layer single-card stack passed five Serving cases in each of HBG eager and explicit Graph. A split prefill candidate also passed TMR, HBG eager, and Graph golden checks. | Lib [#1758](https://github.com/hw-native-sys/pypto-lib/pull/1758) and [#1762](https://github.com/hw-native-sys/pypto-lib/pull/1762) supply eager prefill and its KV golden correction. Explicit prefill MLP recording passed standalone golden on this branch. Per-layer decode recording passed an eight-token Serving golden in HBG eager and Graph. Request-local token feedback passed its standalone state golden and single- and multi-request checks in both HBG modes with the matching Serving review snapshot. |
| A3 DeepSeek V4 Flash MTP K1 | The 43-layer eight-card integrated stack completed a 64-input, 128-output HTTP case with both HBG configurations. Default HBG records selected prefill regions. The explicit Graph configuration also records decode attention. | Port bounded prefill and fused K1 decode into the shared model implementation and validate the Serving ABI and replay boundaries. MTP prefill and the LM head used ordinary HBG in the historical run. |
| A5 DeepSeek V4.1 Flash | Synchronous HBG with TWAIT and AICPU compatibility scheduling passed 40-layer EP4/DP4/TP1 HTTP lifecycle checks at 8,192 input tokens. Two teacher-forced repetitions matched TMR byte-for-byte across 222 corresponding comparison steps for logits and effective KV and compressor state. | Port against current Lib main. Recheck indexer dependencies, routed MoE workspace ownership, and communication waits against the current implementation. Explicit Graph, resident-only scheduling, and whole-model Async await qualification. |

Coverage is limited to the models and configurations listed here.

## A5 V4.1 forward port

The forward-port baseline is Lib `eae7902015d84817707da32034ab8b44039be336`.
Lib [#1763](https://github.com/hw-native-sys/pypto-lib/pull/1763) adds
register indexer and stable TopK implementations. The current C1A decode
entry still calls `score_select_paged`, so validate the active entry when
checking the historical device-count and dependency fixes.

The C1A decode component passed a single-card, 36-row golden under TMR
and HBG on the prepared A5 toolchain. The port schedules the paged
index plan and sparse-attention scratch initialization on the device.
The EP2 full layer and EP4 Serving qualification remain separate.

Start with the current EP2 decode-layer golden under TMR and HBG. Validate
any indexer fix, routed MoE workspace reuse, or communication wait change
against its current callers. Qualify the forward port on the full EP4
model with teacher-forced output and state comparisons and the Serving
request lifecycle.

## PyPTO and Simpler dependencies

As checked on 2026-11-11, PyPTO main
`fc50efe71ab1e6adbfd7703d3f6ac8b209d85c0b` pins Simpler
`355ce3015e3e344d1fc3a61c206ee01823ce292c` and PTOAS v0.72.
Use the PyPTO checkout's runtime gitlink and toolchain pins for ordinary
builds. Record any runtime override with the validation result.

| Capability | Upstream status on 2026-11-11 |
| --- | --- |
| Graph GM pipe capacity, HOST/CHIP returns, and outer completion | PyPTO [#3275](https://github.com/hw-native-sys/pypto/pull/3275), [#3276](https://github.com/hw-native-sys/pypto/pull/3276), and [#3277](https://github.com/hw-native-sys/pypto/pull/3277) merged. |
| Graph communication context origins | PyPTO [#3322](https://github.com/hw-native-sys/pypto/pull/3322) merged at `09e5969a`. |
| Explicit predecessors for Graph invocations | Simpler [#2869](https://github.com/hw-native-sys/simpler/pull/2869) merged at the PyPTO runtime pin above. |
| Graph admission accounting | Simpler [#2930](https://github.com/hw-native-sys/simpler/pull/2930) merged at `612ada38`, after that PyPTO runtime pin. |
| Submitted JIT specialization and parameter directions | PyPTO [#3321](https://github.com/hw-native-sys/pypto/pull/3321) and [#3323](https://github.com/hw-native-sys/pypto/pull/3323) are open. Check each model path against main before declaring either mandatory. |
| Hazard-map growth, ready-queue sizing, and native communication-domain lifetime | Simpler [#2931](https://github.com/hw-native-sys/simpler/pull/2931), [#2932](https://github.com/hw-native-sys/simpler/pull/2932), and [#2934](https://github.com/hw-native-sys/simpler/pull/2934) are open. Treat runtime overrides as case-specific until validated. |

Serving [#308](https://github.com/hw-native-sys/pypto-serving/pull/308)
merged at `6474df7d` and propagates the selected runtime when compiling
Programs. Qwen Program prefill
[#317](https://github.com/hw-native-sys/pypto-serving/pull/317) and KV
preemption recovery
[#312](https://github.com/hw-native-sys/pypto-serving/pull/312) remain open.

## Mainline integration evidence

The following checks used PyPTO `fc50efe7` and Simpler `0569c779` with
PTOAS v0.72. The Simpler main experiment selected an explicit runtime
override because PyPTO still pins `355ce301`.

| A3 model and entry | Result |
| --- | --- |
| Qwen3-14B prefill and decode | Eager and Graph compiled with both the pinned runtime and Simpler main. With Simpler main, each mode passed an eight-token Serving precision test on one A3 card with Lib `f4711ad` and the matching Serving review snapshot. |
| DeepSeek V4 Flash MTP prefill | The candidate compiled main prefill and MTP prefill. These candidate kernels are still being migrated into the shared Lib model. |
| DeepSeek V4 Flash MTP fused decode | The candidate stopped in the PyPTO parser on `pl.submit(_moe_expert_chunk_recorded, ...)` with both Simpler versions. PyPTO [#3321](https://github.com/hw-native-sys/pypto/pull/3321) covers submitted JIT callables. A single-PR comparison is in progress. |

## Tested version combinations

| Validation | PyPTO | Simpler | Lib | Serving |
| --- | --- | --- | --- | --- |
| A3 historical Qwen and DeepSeek integration | `7f7ce3b01` | `90fcb0089` | `fbe9cec70` | `6ac05e563` |
| A3 Qwen split prefill golden, PTOAS v0.72 | `11967c7bf27dfc50f2ebd0ef06e2a847b2253d9a` (#3323 CI merge snapshot) | `355ce3015e3e344d1fc3a61c206ee01823ce292c` | `da5350d2` plus prefill recording | Standalone |
| A3 Qwen branch prefill golden, PTOAS v0.72 | `11967c7bf27dfc50f2ebd0ef06e2a847b2253d9a` | `355ce3015e3e344d1fc3a61c206ee01823ce292c` | `f693535` plus prefill MLP recording | Standalone, TMR/eager/Graph |
| A3 Qwen branch decode golden | `7f7ce3b01019b87c70c5ffbbea4f354c442d7321` | `90fcb00898b58f86bb491dcca8e51b4d4812dc36` | `f693535` plus prefill and decode recording | A3 review snapshot with Graph entry routing, eight exact tokens in each HBG mode |
| A3 Qwen request-local feedback | `7f7ce3b01019b87c70c5ffbbea4f354c442d7321` | `90fcb00898b58f86bb491dcca8e51b4d4812dc36` | Qwen recording commits plus request-local token state | A3 review snapshot with feedback selection and retirement, two native request cases per HBG mode and 14 focused CPU tests |
| A5 V4.1 synchronous HBG | `8a11f4e6` plus snapshot patch | `31f9e72e` plus snapshot patch | `740bc389` plus snapshot patch | `29fed17d` plus snapshot patch |

The A5 synchronous model generated TWAIT. The historical PyPTO and Simpler
snapshots also carried deferred-completion development changes. Record a
fresh toolchain and runtime pairing when qualifying the forward port.

When changing coverage, record the Lib commit, PyPTO and Simpler pins,
Serving commit for integration tests, hardware and parallelism, input and
output lengths, execution mode, and the validation that passed. Keep
performance results tied to their own workload and measurement protocol.
