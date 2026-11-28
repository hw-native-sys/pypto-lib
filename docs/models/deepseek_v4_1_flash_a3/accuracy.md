# Native vLLM-Ascend accuracy comparison

The comparison tools use TP4/DP4/EP16, PP1/PCP1/DCP1 and the original 40-layer
Ascend W8A8 checkpoint. vLLM enables Engram CPU offload and DP shared memory.
The reference uses native model, router, sampler and trace-replay code, without
monkeypatches. TP1/DP16 temporary comparison scripts have been removed.

The tooling provides input preparation, independent captures and CPU comparison.
Configuration checks and CPU tests do not establish complete-network accuracy.
HCCL, weight loading and real Prefill/Decode accuracy require a separate
16-device model run.

## Prepare one immutable case file

In the PyPTO environment, run from the repository root:

```bash
export CHECKPOINT=/path/to/DeepSeek-V4.1-Flash-w8a8
export RUN_DIR="$PWD/build_output/deepseek-v41-accuracy"
python -m tools.deepseek_v41_accuracy prepare \
    --checkpoint "$CHECKPOINT" \
    --cases tools/deepseek_v41_accuracy.example.json \
    --output "$RUN_DIR/cases.json"
```

Both collectors must use this same prepared file. Preparation freezes the
explicit token IDs, vocabulary, topology and checkpoint location. It hashes
the configuration, quantization description, safetensors index and tokenizer,
and records each indexed weight shard's size and modification timestamp.
This detects metadata or shard-stat changes; it does not hash every weight byte.
Prepared cases and capture/report outputs refuse overwriting existing results.

The source file contains a `cases` list. Each case needs a unique `id`,
`prompt_token_ids` and `trace_token_ids`. Alternatively, replace either token
list with `prompt` or `trace` text. Preparation tokenizes text once using the
checkpoint tokenizer with `add_special_tokens=False`; neither collector
retokenizes the frozen IDs or applies a chat template. Supply fully formatted
chat token IDs when testing chat inputs.

The example's trace is a fixed history, not an expected answer. Each prompt
and trace must be nonempty, the trace may contain at most 256 IDs, and their
combined length must not exceed 16384 positions.

## Align Prefill and Decode

For prompt `P` and trace `[t0, t1, t2]`, both captures contain:

| Step | Phase | Model input history | Token whose score is reported |
| --- | --- | --- | --- |
| 0 | Prefill | `P` | `t0` |
| 1 | Decode | `P + [t0]` | `t1` |
| 2 | Decode | `P + [t0, t1]` | `t2` |

A trace of length N collects N full-vocabulary distributions: one final
Prefill distribution and N-1 actual one-token Decode distributions. The final
trace token is scored but never consumed as input. Top1 and TopK come from
the raw distribution, independently of the forced trace token.

The PyPTO collector launches `models.deepseek_v4_1_flash_a3.generate` for
each case. Prefill consumes the prepared prompt. Decode receives the fixed
trace tokens while both phases share the request's weight and cache allocations.
The collector verifies the actual Decode input IDs and the number of calls in
each phase before saving the captured distributions.

The vLLM collector sets `enable_trace_replay=True`,
`SamplingParams(trace_decode_token_ids=..., temperature=0.0, logprobs=-1)` and
`VLLM_USE_V2_MODEL_RUNNER=1`. The engine also sets `max_logprobs=-1` and
`logprobs_mode="raw_logits"`. The default limit of 20 is insufficient for a
full-vocabulary comparison. vLLM keeps its `Logprob.logprob` field name in
raw-logits mode; this collector stores that field as logits, then computes
normalized log probabilities during CPU comparison. `prompt_logprobs` is
incompatible with trace replay and is not requested.

The collector sets `max_tokens` to the frozen trace length explicitly. Native
request normalization can otherwise truncate a longer trace at the sampling
parameter's default token limit.

Only DP rank 0 receives a request. Other DP engines participate through native
dummy batches for expert-parallel collectives. All TP ranks in DP rank 0
receive the same IDs. Prefix caching, asynchronous scheduling and speculation
are disabled; the default Prefill chunk size is 16 in both collectors.

## Inspect configuration without devices

In the PyPTO environment:

```bash
python -m tools.deepseek_v41_accuracy pypto \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/pypto" --check-config
```

In the vLLM-Ascend environment, with this repository on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
ASCEND_RT_VISIBLE_DEVICES='' python -m tools.deepseek_v41_vllm_reference \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/reference" --check-config
```

The vLLM check constructs its native `EngineArgs` configuration and validates
trace replay and full-vocabulary sampling requests. It creates no workers,
loads no model weights and verifies that the NPU remains uninitialized.
Hide devices for this check because native plugin imports can otherwise probe
Triton device properties and initialize the NPU. Apply the empty visibility
only to the configuration command; capture uses the scheduler's allocation.
The installed vLLM and vLLM-Ascend sources must both support V4.1, Engram,
and native trace replay. A successful smoke test with another model or a
matching package version alone does not establish these capabilities. An
unsupported `engram_config` or `enable_trace_replay` argument is an environment
mismatch; do not remove those settings to obtain a reference capture.

## Capture on allocated devices

Run each collector separately through the site's task scheduler with 16
allocated devices exported as comma-separated physical IDs in `TASK_DEVICE`.
An existing `ASCEND_RT_VISIBLE_DEVICES` must match that mapping. Use the same
`--chunk-size` for both captures. The following commands belong inside those
allocated jobs, after activating each corresponding environment.

Native vLLM-Ascend:

```bash
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    --module tools.deepseek_v41_vllm_reference \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/reference" --chunk-size 16
```

The external launcher is required: a single-process offline `LLM` does not
support this four-way data-parallel reference configuration.

PyPTO:

```bash
python -m tools.deepseek_v41_accuracy pypto \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/pypto" --chunk-size 16
```

Pass `--graph-dir "$GRAPH_DIR"` to the PyPTO collector to use an existing
pair of compiled graphs. The Prefill capacity must cover the requested chunk
size and the history capacity must cover every prompt plus its trace. With no
graph directory the collector builds the pair once for the prepared cases.

Each capture writes a manifest and one FP32 `[trace_length, vocabulary_size]`
tensor per case. The manifest records checkpoint identity, case-file hash,
exact histories, phase labels and tensor hashes. Interrupted or failed
captures cannot be compared as completed results.

## Compare on CPU

In the PyPTO environment, first measure without imposing an acceptance policy:

```bash
python -m tools.deepseek_v41_accuracy compare \
    --actual "$RUN_DIR/pypto" --reference "$RUN_DIR/reference" \
    --output "$RUN_DIR/metrics.json" --topk 10
```

The report contains per-step maximum/mean/p99 absolute error, RMSE, relative L2,
cosine similarity, log-probability error, KL divergence, Top1 agreement, TopK
overlap and trace-token log probabilities. Errors and quantiles are computed
in FP64; quantiles use linear interpolation. Each step records the worst
vocabulary coordinate, both logits and their signed difference. Native
Top1-to-Top2 margins retain ties as zero, including when `--topk 1` is used.
A single-token vocabulary has no Top2 margin. Reference norms below `1e-12`
are flagged, and relative L2 keeps its `1e-12` denominator floor.

The overall, per-phase and per-case summaries contain the mean, p95 and
maximum of step RMSE, relative L2 and p99 absolute error. Each summary also
identifies its worst RMSE, relative-L2 and maximum-absolute-error step by case,
phase, step and context length. The attached vocabulary coordinate is the
largest absolute error within that step. Overall means weight every step
equally; `by_case` preserves short cases and their phase summaries. Chunk size
remains explicit in each report, and the report hashes both source manifests
so baseline and candidate reports can be tied to the same native capture.

The tool rejects differing inputs, histories, checkpoints,
chunk sizes, incomplete vocabularies, modified artifacts and nonfinite data.

Measure baseline run-to-run variation with two independently executed captures
from the same backend, frozen cases, environment and chunk size:

```bash
python -m tools.deepseek_v41_accuracy repeatability \
    --backend pypto --first "$RUN_DIR/pypto-repeat-1" \
    --second "$RUN_DIR/pypto-repeat-2" --output "$RUN_DIR/pypto-repeatability.json"
```

Use `--backend vllm-ascend` for native repeats. This command shares the input,
history and tensor-integrity checks and reports per-step, per-phase and per-case
variation. Its `actual` fields refer to the first run and `reference` fields to
the second; it always reports `measured`, without accuracy gates. The reports
measure observed variation between those runs, not a statistical bound on future
runs. Record and inspect both runs' full environment identities separately.
The ordinary `compare` command still requires PyPTO against native vLLM-Ascend;
do not relabel manifests to compare repeats or candidates.

Without an explicit acceptance policy the status is `measured`, even when
all values match. To obtain a `pass` or `fail` verdict, supply the agreed
`--rtol` and `--atol` together, optionally `--max-error-ratio` and
`--require-top1`. The logits gate reuses `golden.ratio_allclose`, defaults to
zero allowed outliers, and applies to every step. A failed gate exits with
code 1; invalid input exits with code 2. `--require-top1` can also be used
alone when Top1 agreement is the complete acceptance criterion.

CPU regressions for the tools run without NPU allocation:

```bash
python -m pytest tools/test_deepseek_v41_accuracy.py -q
```
