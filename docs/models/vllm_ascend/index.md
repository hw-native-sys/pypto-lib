# vLLM Ascend CSA

PyPTO-Lib provides the DeepSeek V4 Flash CSA fused operator in
`models/vllm_ascend/deepseek_v4_flash/csa/`. It supports SPMD and MPMD execution,
standalone numerical validation, and vLLM Ascend integration through native
weight, metadata, cache, and output bindings.

## Implementation and layout

Operators are grouped by model under `models/vllm_ascend/`. The CSA package
contains the compute implementations, shared host bindings, Torch reference,
and validation entries.

```text
models/vllm_ascend/
`-- deepseek_v4_flash/
|   |-- config.py                  # Model dimensions and configuration
|   `-- csa/
|       |-- __init__.py            # CSA implementation selection
|       |-- service.py             # Shared TND service lifecycle
|       |-- common/                # Weights, storage, contracts and shared kernels
|       |-- spmd/                  # Single-SPMD entry, scheduling and stages
|       |-- mpmd/                  # Multi-task entry, dependencies and stages
|       |-- fixtures.py            # Seeded TensorSpec inputs and Native cache pages
|       |-- reference.py           # Independent Torch fused-operator golden
|       |-- validate.py            # Standalone SPMD/MPMD validation
|       |-- serving_fixture.py     # Real-weight service fixture
|       |-- validate_serving.py    # Loader and service integration validation
|       `-- legacy/                # Compatibility performance mode
```

| Package under `deepseek_v4_flash.csa` | Purpose | Entry point |
| --- | --- | --- |
| `spmd` | Single SPMD CSA | `entry.decode_csa_spmd` |
| `mpmd` | MPMD CSA | `entry.decode_csa_tnd` |
| `common` | Shared weights, Native storage adapter, graph contracts, configuration and kernels | No serving entry |
| `legacy` | Compatibility `performance` mode | `decode_csa._decode_csa_tp1_layer` |

Both mode-specific services inherit `csa.service.TNDServiceRuntime`.
Each mode supplies its own weight-root signature, policy and prepared binding.
Shared code does not import either
mode or the legacy implementation. Model configuration lives one level above
CSA; execution capacities and numerical constants remain in `csa/common`.
SPMD and MPMD organize their computation stages and scheduling separately.

Both TND services expose
`CSAServiceRuntime(attention, operators, max_num_seqs, layer=None)`, with
`operators=None`. They provide eligibility checks, Native fallback, cache
connector notifications, graph signatures, stable workspaces, and
in-place cache/output ownership. The prepared binding includes mHC pre,
input normalization, CSA, and mHC post; it does not include the MoE half-layer.

Python files in this integration use the repository's CANN Open Software
License Agreement Version 2.0 header template.

## Connect vLLM Ascend

The integration targets vLLM Ascend with vLLM 0.25.1rc1 and requires
the `PyptoCSADeepseekV4ForCausalLM` model hook, CSA variant selector, and Model
Runner V1 service hooks. Connect these hooks to PyPTO-Lib using the imports below. The library does
not install these hooks into an unmodified vLLM Ascend release. Other versions require
separate adapter validation.

Make the **repository root**, rather than `models/vllm_ascend`, available in
every serving worker's `PYTHONPATH`. PyPTO-Lib uses the existing `models.*`
namespace and does not currently provide an editable-install package.

```bash
export PYPTO_LIB_ROOT=/path/to/pypto-lib
export PYTHONPATH="$PYPTO_LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export PTO_CSA_VARIANT=spmd_tnd  # or mpmd_tnd
```

Update the serving-side integration at these call sites:

| vLLM Ascend call site | PyPTO-Lib interface |
| --- | --- |
| `ops/pypto/variant.py` | Import `csa_package` from `models.vllm_ascend.deepseek_v4_flash.csa`; resolve `performance`, `spmd_tnd`, and `mpmd_tnd` with it |
| `models/pypto_deepseek_v4.py` | Import `validate_configuration` from `csa.common.service_config`, weight layout checks from `csa.common.nz_mode`, and `ATOMIC_ADD` from `csa.common.reduction` |
| `worker/model_runner_v1.py` | Import the CSA configuration and graph replay predicates from `csa.common.service_config` |
| `compilation/acl_graph.py` | For legacy HBG graph replay, import `validate_graph_replay` from `csa.legacy.host_metadata` |

The abbreviated `csa.*` paths in the table start at
`models.vllm_ascend.deepseek_v4_flash.csa`. In the variant selector, replace
the package constants with:

```python
from models.vllm_ascend.deepseek_v4_flash.csa import csa_package

_CSA_PACKAGE = csa_package("performance")
_CSA_TND_PACKAGE = csa_package("spmd_tnd")
_CSA_MPMD_TND_PACKAGE = csa_package("mpmd_tnd")
```

These changes belong in the serving adapter. The operator library does not
modify an installed vLLM Ascend checkout automatically.

Keep the existing launcher, model registration, and
`PyptoCSADeepseekV4ForCausalLM` architecture selection. Set `PTO_CSA_VARIANT` to
`spmd_tnd` or `mpmd_tnd` to select a validated implementation in this library.
An invalid variant fails; a missing library does not silently select a
different kernel.

Inspect the actual production route before launching the server:

```bash
python - <<'PYCODE'
from importlib import import_module
from vllm_ascend.ops.pypto.variant import selected_variant, variant_package
package = variant_package()
print(selected_variant(), package)
print(import_module(package + '.service').__file__)
print(import_module(package + '.entry').__file__)
PYCODE
```

For direct use from another adapter, `models.vllm_ascend.deepseek_v4_flash.csa.csa_package(variant)`
returns the package path without importing Torch or initializing an NPU. Load
its `.service` module inside the serving worker after configuring the device
stack; initialize PyPTO through the existing vLLM model hook.

## Supported integration contract

The TND service supports the following configuration:

- DeepSeek V4 Flash, C4 attention, TP=1 and PP=1; the serving framework owns EP.
- BF16 activations and the audited symmetric W8A8 ModelSlim or
  compressed-tensors weight format.
- NZ2 weights, ND KV caches, 32-token cache blocks, and
  `tensormap_and_ringbuffer` runtime.
- DSpark target decode with up to six query tokens per request, including
  padding and varying actual query lengths; configured speculation uses five
  draft tokens.
- Model Runner V1. The adapter retains eager and `FULL_DECODE_ONLY` metadata
  contracts; the tested PyPTO eager bridge currently rejects graph capture,
  so this bundle is validated in eager mode only. Unsupported per-call
  descriptors use the existing Native fallback.

The descriptor bounds are 8K through 1M history, but those bounds are not a
claim that every shape has been validated. The standard validation case uses
**BS=16, history=131072, Q=6, T=96** on A3 with environment CANN **9.0.0**.
Full-model TP1-EP16, BS=24, 128K execution requires separate end-to-end
validation; single-layer results do not establish that level of coverage.

## Toolchain requirements

The implementation requires a compatible PyPTO, runtime, and PTOAS combination.
The tested compiler baseline is `863c11938ab4` (a historical PyPTO PR #3230
head, **not** an official main commit), runtime `2dd4cbec9a97` (PR #2647 head),
and PTOAS v0.70 baseline
`105e48b046c9`, with local fixes. Required capabilities include:

- PyPTO GM signal operations, barrier/backoff support and correct mixed-kernel
  classification after dead metadata cleanup.
- L0C phi handling and UB memory packing for the SPMD kernel.
- Compiler/runtime agreement on the enlarged argument capacity (256) and
  exact device tensor aliases.
- PTOAS lowering for the signal-wait backoff.

The corresponding upstream capability PRs are
[PyPTO #3252](https://github.com/hw-native-sys/pypto/pull/3252),
[Simpler #2846](https://github.com/hw-native-sys/simpler/pull/2846), and
[PTOAS #1728](https://github.com/hw-native-sys/PTOAS/pull/1728).
The eager execution prerequisites remain
[PyPTO #3194](https://github.com/hw-native-sys/pypto/pull/3194),
the cold-start and source-lease support in
[PyPTO #3230](https://github.com/hw-native-sys/pypto/pull/3230), and
[Simpler #2647](https://github.com/hw-native-sys/simpler/pull/2647), or their
upstream equivalents. These links identify capabilities under review;
they are not a substitute for a qualified combined toolchain release.

Use a toolchain bundle with these capabilities. The recorded results apply
to the versions and local fixes above; an unmodified upstream toolchain or a
newer vLLM Ascend release requires separate validation.

## Run the two validation paths

Run from the PyPTO-Lib root, with the supported toolchain and CANN environment
already configured. Use the site's device allocator and pass its assigned
single device ID. Both validation entries are provided by this repository.

### Standalone fused operator

This path needs Torch, torch_npu, PyPTO and its runtime, but **no model weights,
vLLM, or vLLM Ascend installation**. It runs the same `PreparedDynamicCSA` and
kernel entry used by serving; there is no alternate validation kernel.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ASCEND_ENABLE_NZ=2
python -m models.vllm_ascend.deepseek_v4_flash.csa.validate \
  -p a2a3 -d "$DEVICE" --mode both
```

The default cases use 8K history: uniform BS=1/Q=6; mixed BS=2 with Q=6/5,
different compression boundaries, inactive requests and padded token rows;
and single-token decode with no new compressed row at the default history.
Select `--case uniform|mixed|single` for one case. `--mode spmd|mpmd` selects
one implementation; `both` additionally requires bitwise equality of all nine
observable states. Both implementations execute twice against restored input
caches to check persistent-workspace reuse.

The standard larger case is explicit:

```bash
python -m models.vllm_ascend.deepseek_v4_flash.csa.validate \
  -p a2a3 -d "$DEVICE" --mode both --case uniform \
  --batch 16 --history 131072 --query 6 --save-data \
  --output build_output/csa_b16_128k
```

`fixtures.py` uses `TensorSpec` inputs and seeded nonzero weights/cache contents.
`reference.py` independently computes mHC pre, normalization, query/KV and
compressor projections, Indexer scoring/Top-K, sparse attention, output
projection and mHC post with Torch. `golden.validate_golden` and the repository's
`ratio_reldiff` / `ratio_allclose` comparators enforce numerical accuracy.
The composed output uses the existing CSA gate (`diff_thd=0.004`,
`pct_thd=0.03`, `max_diff_hd=2`). Written cache/state rows and Top-K scores have
separate dtype-aware gates in `compare_outputs`; unchanged cache bytes,
metadata, workspace guards and inactive outputs are checked exactly. Top-K
also checks unique causal indices, selected-score correctness and the cutoff,
allowing equivalent choices near a rounding boundary.

The driver uses the native prepared binding instead of `golden.run` to retain
NZ storage and exact workspace aliases. It does not treat private scratch
buffers as mathematical outputs. This follows the external-runtime driver
pattern in the [compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md).
Numerical comparison to Torch uses tolerances; SPMD/MPMD and repeated-run
comparisons require identical bytes.

`--compile-only` compiles the selected cases without device execution and
reports `accuracy: NOT_RUN`. `--save-data` saves inputs, expected outputs and
device outputs (including a failed comparison) under the output directory.
`--dump-passes` controls compile diagnostics; `--enable-chip-swimlane` configures
the native runtime diagnostics. Direct JIT execution uses `CompileOptions`
and `pypto.torch.init`; compile-only warmup uses `RunConfig`. On the current
toolchain bundle, both validation drivers also call the internal eager-state
`drain()` after device synchronization: Torch synchronization alone does not
retire the native run handle before another variant is registered. This is a
validation lifecycle dependency, not a change to kernel computation. Remove
it when the supported public bridge owns completed-ticket retirement.
Successful execution ends with `[RUN] PASS` and `result.json` containing
`passed: true` and `accuracy: PASS`. A failed case returns a nonzero exit code.
Compilation success alone is not numerical validation. Simulator execution is
not claimed by this driver.

### vLLM Ascend integration

Connect the serving hooks described above, then use the real compressed-tensors
checkpoint and the existing vLLM environment:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PTO_CSA_RING_HEAP_MB=256
export PTO_CSA_RING_TASK_WINDOW=16384
python -m models.vllm_ascend.deepseek_v4_flash.csa.validate_serving \
  --model /path/to/deepseek-v4-flash-checkpoint -d "$DEVICE" \
  --mode both --batch 16 --history 131072 \
  --output build_output/csa_serving_b16_128k
```

This entry loads actual layer-4 attention and mHC weights with the vLLM loader,
checks both weight adapters and NZ format, builds Native metadata/cache views,
and resolves each service through the production `variant_package()` selector.
It requires service eligibility and checks cache guards, read-only metadata,
repeated-call identity and SPMD/MPMD byte identity. `result.json` must contain
`passed: true`; the nine states are saved in each mode's attention-state file.
It validates the serving adapter and single-layer equivalence, not a Native
numerical golden or full-model generation. TP1-EP16, BS=24/128K, basic question
accuracy and graph replay remain separate end-to-end acceptance tests.

## Validation entry regression

Both repository-owned entries passed on A3 / CANN 9.0.0 with the toolchain
bundle above on 2026-11-02:

| Entry / case | Shape | Result |
| --- | --- | --- |
| Standalone uniform | BS=1, history=8192, Q=6 | Both modes passed Torch, guards and repeat checks; nine states bit-identical across modes |
| Standalone mixed and padded | 2 active requests, histories=8192/8195, Q=6/5; T capacity=24, request capacity=5 | Both modes passed; inactive outputs masked; one/two compressed-row boundaries covered |
| Standalone single-token | BS=1, history=8192, Q=1 | Both modes passed; no new compressed cache row; nine states bit-identical |
| Standalone standard case | BS=16, history=131072, Q=6, T=96 | Both modes passed Torch, guards, repeat and cross-mode byte checks |
| Real-weight serving | Layer 4, BS=16, history=131072, Q=6 | Both modes passed loader/adapter, cache guards and repeat checks; nine states bit-identical across modes |
| Host-only serving contracts | No NPU execution | 23 passed |

Each successful device run exited with code zero and wrote `result.json` with
`passed: true`. Standalone reports also record `accuracy: PASS`,
`mpmd_spmd_bit_equal: true` and `standalone_without_vllm: true`. The standalone
reference uses numerical tolerances; byte equality applies to the two device
implementations and repeated calls, not to Torch versus device results.

The nine compared states are `x_out`, `idx_topk`, `swa.0`, `compressed.0`,
`state.0`, `indexer.0`, `indexer.1`, `indexer_state.0`, and `idx_topk_scores`.
These entry-level results do not establish full-model generation, multi-card
EP16, device graph capture/replay, or the compatibility `performance` mode.
They are correctness checks, not performance measurements.

## Host-only contract tests

Host-only serving contracts run without a working NPU stack:

```bash
python -m pytest tests/contract/test_vllm_ascend_csa_contract.py -q
ruff check models/vllm_ascend tests/contract/test_vllm_ascend_csa_contract.py
```

The tests cover mode resolution and invalid choices, lightweight namespace
imports, descriptor fallback, graph signature invalidation and quantization
rejection. Device acceptance additionally runs the standalone Torch comparison
and loads actual finalized model weights through vLLM to execute both services
through `variant_package()`. Numerical tolerance against Torch and cross-mode
bitwise equality of the nine output/cache states are separate requirements.
