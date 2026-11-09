# KDA core precision and performance comparison

`models/glm5_3_flash/kda_benchmark.py` runs prefill and decode as separate
workloads. It freezes one set of inputs and initial states for PTO3 and the
AscendC KDA operators used by vLLM Ascend. The reference environment needs
`torch_npu`, `vllm`, `vllm_ascend` and `fla_npu`; ordinary PTO3 CI does not.

## Workload and measurement boundary

Both implementations consume BF16 Q/K/V, preprocessed FP32 log-decay and beta,
and FP32 recurrent state in `[V, K]` order. K and V are both 128. TP8 uses
eight local heads; TP16 uses four. Each case uses one device and no collective.

The timed boundary includes Q/K normalization and the delta-rule computation.
Projection, convolution, gate/beta preprocessing, input transfers, state-layout
conversion and reset copies are excluded. Decode calls the production
`recurrent_kda` through its standalone entry with prepared request metadata.
Prefill calls the existing `prefill_kda_test` entry.

PTO3 prefill retains chunk16 and its existing BF16 rounding boundaries.
The vLLM Ascend reference uses its AscendC chunk64 kernel and BF16 normalized
prefill operands; decode normalizes inside its recurrent kernel. These
implementation differences remain visible in the precision report.

PTO3 reads immutable seed rows and writes disjoint working rows. The reference
resets its private state and input buffers before each timed call. Those copies
precede the start event. Thus every sample starts from the same logical state.
Reference graph mode records timing events outside graph capture. Eager mode is
available explicitly with `--vllm-mode eager`; the report records which was used.

## Run a paired case

Run from the repository root with the appropriate environment active for each
engine. Both environments must see the same case directory; when using a
container, mount that directory and this script. No model checkpoint is needed.

Prepare inputs, both CPU references and native PTO3 compilation without a device:

```bash
python models/glm5_3_flash/kda_benchmark.py \
  --engine pto --phase decode --tp 8 --ep 8 --requests 4 --tokens 4 \
  --case-dir build_output/kda-decode-b4-tp8 --compile-only
```

Execute each engine through the host's device allocator. On hosts with
`task-submit`, activate the selected engine environment before submitting:

```bash
task-submit --device auto --device-num 1 --max-time 1800 \
  'python models/glm5_3_flash/kda_benchmark.py --engine pto \
   --case-dir build_output/kda-decode-b4-tp8 -d "$TASK_DEVICE"'
```

In the vLLM Ascend environment, with the same frozen directory:

```bash
task-submit --device auto --device-num 1 --max-time 1800 \
  'python models/glm5_3_flash/kda_benchmark.py --engine vllm-ascend \
   --case-dir build_output/kda-decode-b4-tp8 -d "$TASK_DEVICE"'
```

Use the same physical device and sampling counts for a controlled comparison.
Defaults are 100 measured calls and 10 warmups, matching the daily performance
manifest. The script records every sample and the measurement convention.

Generate the comparison in the PTO3 environment; this step is CPU-only:

```bash
python models/glm5_3_flash/kda_benchmark.py --engine report \
  --case-dir build_output/kda-decode-b4-tp8
```

For prefill, prepare another directory with `--phase prefill --requests 1
--tokens 8192`, or `--requests 4 --tokens 128`. Decode comparison is one token
per request. The operator's separate `--component verify` entry covers MTP
convolution, recurrent checkpoints and accepted-state restoration.

## Reports and validation

The input manifest contains the workload and an SHA256 digest. Every engine
must match it. `comparison.md` and `comparison.json` report:

- each engine versus the shared FP32 recurrence, and PTO3 versus the reference;
- output and final working-state relative L2 percentage and maximum absolute error;
- strict allclose results with `rtol=0.02`, `atol=0.02` and zero tolerated outliers;
- median, mean, standard deviation and all raw timing samples;
- source hashes, dependency versions, state layout and reference execution mode.

PTO3 additionally validates against its original implementation-aligned golden:
the mixed-precision chunk reference for prefill and FP32 recurrence for decode.
Read-only seed and unused state rows must remain exact. The shared FP32
reference is a separate mathematical diagnostic; the report does not replace
the existing PTO3 golden or hide differences between chunk algorithms.

Nonfinite reference results fail immediately. A numerical comparison failure
still writes the report and returns a nonzero exit status. Simulator runs can
validate PTO3 numerics but do not produce device performance numbers.

## Daily performance cases

`.github/scripts/perf_cases.json` registers versioned PTO3 cases for TP8 and
TP16: decode with 1/4/32 requests, prefill with 1x128, 4x32 and 1x8192 tokens,
and separate MTP verification cases. The v2 profile fixes the core boundary and
immutable-state policy. The old `kda-verify-b32q6` ID is renamed to
`kda-prefill-b32q6`, reflecting the prefill path it actually executes; its
arguments and numerical workload are unchanged.

The comparison backend is optional and is run explicitly in the reference
environment. It does not add a vLLM dependency to the PTO3 CI environment.
