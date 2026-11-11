# HBG model compatibility

This page tracks the HBG model paths on the Lib `dev/hbg` branch. A
historical four-repository result and validation of the current branch are
recorded separately. Graph means explicit recording and replay. Async
qualification covers request state, completion, and retirement in Serving.

| Platform and model | Historical validation | Current branch coverage |
| --- | --- | --- |
| A3 Qwen3-14B BF16 | The integrated 40-layer single-card stack passed five Serving cases in each of HBG eager and explicit Graph. A split prefill candidate also passed TMR, HBG eager, and Graph golden checks. | Lib [#1758](https://github.com/hw-native-sys/pypto-lib/pull/1758) and [#1762](https://github.com/hw-native-sys/pypto-lib/pull/1762) supply eager prefill and its KV golden correction. Explicit prefill MLP recording passed standalone golden on this branch. Per-layer decode recording passed an eight-token Serving golden in HBG eager and Graph. Request-local token feedback passed its standalone state golden and single- and multi-request checks in both HBG modes with the matching Serving review snapshot. |
| A3 DeepSeek V4 Flash MTP K1 | The 43-layer eight-card integrated stack completed a 64-input, 128-output HTTP case with both HBG configurations. Default HBG records selected prefill regions. The explicit Graph configuration also records decode attention. | Bounded prefill, fused K1 decode, and explicit Graph entry code are on this branch. The updated source passed the 43-layer HTTP check in eager and explicit Graph. MTP prefill and the LM head used ordinary HBG in the historical run. |
| A5 DeepSeek V4.1 Flash | Synchronous HBG with TWAIT and AICPU compatibility scheduling passed 40-layer EP4/DP4/TP1 HTTP lifecycle checks at 8,192 input tokens. Two teacher-forced repetitions matched TMR byte-for-byte across 222 corresponding comparison steps for logits and effective KV and compressor state. | Port against current Lib main. Recheck indexer dependencies, routed MoE workspace ownership, and communication waits against the current implementation. Explicit Graph, resident-only scheduling, and whole-model Async await qualification. |

Coverage is limited to the models and configurations listed here.

## A5 V4.1 forward port

The first decode ports used Lib `eae7902015d84817707da32034ab8b44039be336`.
The C2A prefill port brings in Lib main `f3427958fbf81afe974b208980ff49d0838c83a4`.
Lib [#1763](https://github.com/hw-native-sys/pypto-lib/pull/1763) adds
register indexer and stable TopK implementations. The current C1A decode
entry still calls `score_select_paged`, so validate the active entry when
checking the historical device-count and dependency fixes.

The C1A decode component passed a single-card, 36-row golden under TMR
and HBG on the prepared A5 toolchain. The port schedules the paged
index plan and sparse-attention scratch initialization on the device.
The C1A reuse golden also passed under both runtimes after moving the
dense-versus-sparse decision to device-side predicates. The EP2 layer 1
golden passed under TMR and HBG with the prepared toolchain. C2A prefill
passed TMR and HBG golden at TP1/DP2 with 48 rows per DP group and six
requests in the long fixture. Score routing now uses device predicates,
and the golden passes the orchestration token count from CPU. Full EP4
Serving qualification remains open.

The TP1 final LM head passed logits and sampled-ID goldens with zero
and eight active rows under TMR and HBG using PyPTO `fc50efe7`, Simpler
`0569c779`, and PTOAS v0.72. Row indices now stay on the device. The
all-padding case uses the regular projection and sampling path.

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
| DeepSeek V4 Flash MTP fused decode | The candidate stopped in the PyPTO parser on `pl.submit(_moe_expert_chunk_recorded, ...)` with both Simpler versions. PyPTO [#3321](https://github.com/hw-native-sys/pypto/pull/3321) covers submitted JIT callables. With only #3321 head `c295a9e1` merged into that PyPTO main, the fused decode compiled in eager and explicit Graph. This establishes the parser dependency for the candidate. The full request is tracked separately. |

## A3 DeepSeek native integration

The updated Lib DeepSeek model subtree
`8646694d8364698fa517539f6632fef709c0b49b` passed the native eight-card,
43-layer fused K1 HTTP test in HBG eager and explicit Graph. Both runs
used 64 prompt tokens, generated 128 tokens, returned identical text,
and shut down normally. This run used PTOAS v0.70, PyPTO
`7f7ce3b01019b87c70c5ffbbea4f354c442d7321`, and Simpler
`90fcb00898b58f86bb491dcca8e51b4d4812dc36`. The Serving review tree
`521089e0aaeecc40324fbff9de401804c7ca8f29` is based on `979bfa25`
with the matching runtime selection and bounded-entry routing. The
review tree also uses the shared Lib request-planning contract. The
PyPTO main plus #3321 combination has separate offline evidence.

The companion Serving patch also replays on Serving main
`6474df7d42e9342c8b74613a8258bf6776f71713`. The resulting Serving tree
`e19166cf766b09ec89d0f7d3e9c9893ce8275c0c` passed 156 targeted
DeepSeek, Qwen request-state, and CLI tests with Lib `b0945fe`. The
CLI preserves the selected chip runtime in the model configuration. The
DeepSeek executor uses that value for code generation, matching Serving
#308's compile-context checks.
Use the reviewed Serving patch for this tree. Its SHA-256 is
`0172718dd53fd2ba3ea7e6248ad33d69786feec79d9486aff1f36c7372bce14a`.
The eight-card validation with this Serving tree and the PyPTO main
plus #3321 compiler is queued separately.

### Start the A3 DeepSeek server

From the matching Serving checkout, point `PYPTO_LIB_ROOT` at Lib
`dev/hbg`. Use the hardware and toolchain from the native result above.
Set `MODEL_DIR` to the V4 Flash MTP W8A8 checkpoint and `DEVICES` to
the eight comma-separated device IDs allocated by your scheduler.

```bash
export PYPTO_LIB_ROOT=/path/to/pypto-lib
python -m pypto_serving.cli.main \
  --model "$MODEL_DIR" --served-model-name dsv4-flash-w8a8 \
  --backend npu --platform a2a3 --devices "$DEVICES" \
  --pypto-runtime host_build_graph --dp 8 --ep 8 --block-size 128 \
  --max-model-len 260 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --npu-memory-utilization 0.90 --long-prefill-token-threshold 1024 \
  --num-speculative-tokens 1 --no-enable-prefix-caching \
  --ring-dep-pool 16384 --ring-task-window 262144 --ring-heap 1073741824 \
  --host 127.0.0.1 --port 8000
```

Append `--hbg-graph` to select explicit Graph decode. Check
`/health` before requesting a completion. The native test uses the
64-input, 128-output case, validates the returned usage, and checks
that the worker shuts down.

## Reproducing the component checks

Use the Lib branch with the [installation workflow](../get-started/installation.md).

For A3 DeepSeek fused decode, build PyPTO from
`fc50efe71ab1e6adbfd7703d3f6ac8b209d85c0b` with the changes from
`c295a9e170bbb3613b19797304352dadf3791ad4`. The latter is the tested
head of PyPTO #3321. Fetch it from `refs/pull/3321/head` in the PyPTO
upstream repository and confirm the SHA before merging it locally.
The offline probe selected Simpler `0569c779` explicitly and used PTOAS v0.72.
Full-model serving also needs the matching bounded prefill and fused
decode routing in Serving.

For A5 C1A reuse, Lib `32c9860` passed both TMR and HBG golden with PyPTO
`fc50efe7`, Simpler `0569c779`, and PTOAS v0.72. After setting up that
pairing, run from the Lib root on an allocated A5 card. Set the device ID
to the scheduler allocation.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
device=0
entry=models/deepseek_v4_1_flash/decode_c1a_reuse.py
python "$entry" -p a5 -d "$device" --start-pos 8192,4096 --query-lens 1,2 --case ragged
python - "$entry" -p a5 -d "$device" --start-pos 8192,4096 --query-lens 1,2 --case ragged <<'PY'
import runpy
import sys
from pathlib import Path

from pypto.pypto_core import passes

entry = Path(sys.argv[1]).resolve()
sys.argv = [str(entry), *sys.argv[2:]]
sys.path.insert(0, str(entry.parent))
with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
    runpy.run_path(str(entry), run_name="__main__")
PY
```

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
