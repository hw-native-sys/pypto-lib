# DeepSeek V4.1 Flash

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Resident CED prefill on A5

[prefill_fwd.py](../../../models/deepseek_v4_1_flash/prefill_fwd.py) composes the
native operators into one L2 invocation per A5 rank. CED (causal encoder-decoder)
runs encoder layers 0–19 over the prompt, publishes layer-20 global KV/index
keys, and runs decoder layers 20–39 over the final causal window. Decoder rows
retain their original positions. Final normalization, the vocabulary head and
greedy sampling also execute on the device.

Supported layouts are **TP1 / EP4 / DP4** (default) and **TP1 / EP8 / DP8**.
Each rank processes one prompt and owns 96 or 48 routed experts per layer.
Attention and shared-expert weights are replicated. The checkpoint reader and
resident packing live in [utils.py](../../../models/deepseek_v4_1_flash/utils.py).
Original checkpoint payloads and scales are reordered without requantization;
all 40 layers' compute weights remain resident across generation steps. Weight
banks use about 75.22 GiB per rank with EP4 or 41.60 GiB with EP8, plus the BF16
vocabulary head, activations, caches and runtime memory.

The entry follows the official BF16/MXFP8/MXFP4 execution policy. FP32 is used
where native operators require it, including reductions, mHC coefficients and
logits; there is no alternate FP32 policy. **Engram is disabled** in both device
execution and host preparation. No Engram tables or hash files are needed, and
any numerical reference must also disable Engram (`engram_layer_ids=()`).

### Compile before allocating devices

Run from the repository root in the environment used for device execution:

```bash
PYTHONPATH=models/deepseek_v4_1_flash:. python - --tp 1 --ep 8 --dp 8 --capacity 128 <<'PY'
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig
from pypto.runtime.distributed_runner import _assemble_chip_callables
from prefill_fwd import RANKS, l3_prefill_fwd

compiled = l3_prefill_fwd.compile(config=RunConfig(
    platform="a5",
    ring_heap=(2 * 1024**3,) * 4,
    distributed_config=DistributedConfig(device_ids=list(range(RANKS)), num_sub_workers=0),
))
_assemble_chip_callables(compiled)
print(compiled.output_dir)
PY
```

The pinned PyPTO version uses this internal assembly helper to finish the
binaries on the CPU. `compile()` generates sources; `prepare()` also starts
device workers. Keep CPU compilation and assembly outside device allocations.
See the [compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md).

### Generate from the released checkpoint

Pass the printed artifact directory with the same capacity and parallel layout.
The checkpoint must include its tokenizer and official `encoding/` files.
On a shared host, submit the run through the device allocator:

```bash
task-submit --device auto --device-num 8 --max-time 1200 \
  'PYTHONPATH=. python models/deepseek_v4_1_flash/prefill_fwd.py \
    --checkpoint /path/to/DeepSeek-V4.1-Flash \
    --runtime-dir /path/to/compiled-artifact \
    --tp 1 --ep 8 --dp 8 --capacity 128 --max-new-tokens 8 \
    --devices "$TASK_DEVICE" --report build_output/ced-generation.json'
```

Use EP4/DP4 and four devices for EP4. `--dp` defaults to `--ep` and must match it;
`--tp` must be 1. The default four prompts are repeated twice with EP8, or supply
exactly one `--prompt` per rank. Prompt length plus `--max-new-tokens` must fit
the compiled capacity, which must be a multiple of 16.

Each new token recomputes both stages over the full prefix. The runner creates
fresh caches and zeroed communication windows (`persistent=False`), with layer
epochs 1–40 and head epoch 1. It does not retain incremental decode state.
The report records token IDs, text, EOS status and execution time. Finite logits
and agreement between device sampling and copied-logit argmax validate execution;
they do not establish low-error whole-model numerical parity. The official mixed
precision path has a known whole-model accuracy gap tracked in
[issue #1394](https://github.com/hw-native-sys/pypto-lib/issues/1394).

This checkpoint-dependent driver is validated on A5 through the dedicated
workflow above. Generic CI sweeps select standalone operator validation entries;
they exclude checkpoint drivers, shared helpers and `_draft.py` files.
