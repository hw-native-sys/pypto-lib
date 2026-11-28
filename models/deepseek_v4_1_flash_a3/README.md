# DeepSeek V4.1 Flash A3

The default text-generation path runs two handwritten PyPTO programs on 16 A3
devices (DP4, TP4, EP16). Each program owns its layer schedule and calls the
phase-specific operators beside the forward entries. One request runs in one TP group.
All sixteen ranks participate in expert communication.

| | Prefill | Decode |
| --- | --- | --- |
| Distributed entry | `prefill_fwd.l3_prefill_fwd` binds rank-local slices and communication windows | `decode_fwd.l3_decode_fwd` binds Decode views, valid counts and windows |
| Per-rank schedule | `l2_prefill_fwd` iterates the requested layer range, normally 0 through 39 | `l2_decode_fwd` runs `decode_encoder` (0–19) and `decode_decoder` (20–39) |
| Attention | Explicit modes 0–5 select SWA, C2A full, C2A reuse, C1A reindex across history, C1A reindex using candidates, and C1A reuse | The encoder and decoder call Decode-specific SWA, C2A and C1A kernels |
| MoE | The compile-time MoE slab ranges from 64 to 2048 rows per TP rank. The graph routes to EP16, synchronizes dispatch and combine, and reuses its temporary slabs across rounds and layers | Graph-local expert dispatch and combine use the Decode valid counts and resident expert weights |
| Input | Prepared embedding and Engram rows, positions, page tables, weights and typed cache tensors | Hidden and pre-mix rows, metadata, weights and views of the same caches. Single-request generation compiles a 16-row input allocation per active rank. The general Decode artifact has 80 rows |

The L3 signatures expose 87 tensor parameters for Prefill and 74 for Decode.
Each phase has its own rank loop. Each rank owns 24 routed experts. Both
phases share the W8A8 checkpoint bank and seven mutable allocations per rank:
window KV payload and scales, compressed KV payload and scales, index payload
and scales, and compressor state. Decode flattens the source and page axes as
views of the Prefill allocations. The final active Prefill row supplies the
first generated token. Each subsequent token takes one Decode call.

## Compile and run

From the repository root, compile into fresh directories. Prefill capacity
must be a multiple of 16 in 16..8192. Both phases need the same history
capacity, including reserved output positions.

```bash
model=models.deepseek_v4_1_flash_a3
python -m "$model.prefill_fwd" --compile-only --capacity 128 \
  --context-capacity 256 --output "$GRAPH_DIR/prefill"
python -m "$model.decode_fwd" --compile-only \
  --context-capacity 256 --output "$GRAPH_DIR/decode"
```

Run generation in a 16-device allocation. The driver checks the artifact
manifests, loads the checkpoint once per session, and prepares the tensor
inputs for each call.

```bash
python -m "$model.generate" --checkpoint "$CHECKPOINT" --devices "$TASK_DEVICE" \
  --graph-dir "$GRAPH_DIR" --tokens 42,43 --max-new-tokens 2 --ignore-eos \
  --output "$RUN_DIR"
```

Use `--prompt` for text or `--tokens-file` for a JSON array of token IDs.
Omit `--graph-dir` to compile both phases under a fresh `--output` directory.
Generation builds both programs with startup communication-window reservation. It
acquires the two 16-rank domains after loading the checkpoint and reports their
times in `window_reservation_seconds`. The first request then uses the retained
windows. For reusable generation artifacts, add `--prepare-windows` to both
`--compile-only` commands. Standalone compilation defaults to the tensor-only ABI.
`generation/tokens.json` holds the generated IDs. `--dump-logits` saves the
vocabulary vector for each emitted token.

Running `python -m models.deepseek_v4_1_flash_a3.prefill_fwd -d 0` compares the
layer-20 replay gather against CPU indexing on one device. It covers layer ranges
starting at 0, 19 and 20. Pass `--compile-only` to compile the full forward graph.

For the bundled 8192-token replay, build Prefill with `--capacity 8192`
and `--prefill-moe-slab-rows 2048`, Decode with `--decode-capacity 16`, and both
programs with `--context-capacity 8200`. Run the compiled pair with

```bash
python -m "$model.prefill_fwd" --checkpoint "$CHECKPOINT" --devices "$TASK_DEVICE" \
  --runtime-dir "$GRAPH_DIR/prefill" --decode-runtime-dir "$GRAPH_DIR/decode" \
  --input-ids-json models/deepseek_v4_1_flash_a3/input_ids.json \
  --prefill-moe-slab-rows 2048 --chunk-size 8192 --max-new-tokens 8 --ignore-eos --output "$RUN_DIR"
```

## Validate logits

The accuracy collector uses the same Prefill and Decode entry points as generation.
It feeds the prepared trace token into each Decode step and records full-vocabulary
FP32 logits. See the [fixed-trace comparison](../../docs/models/deepseek_v4_1_flash_a3/accuracy.md)
and the [module map](MODULE_ALIGNMENT.md).

## Packed request execution

Compile both phases with the same `--request-slots-per-dp` and `--context-capacity`.
Eight slots per DP owner accommodate 32 live requests across DP4. The token
capacity bounds each dispatch on each owner. Cache allocations reserve the
configured context for every slot. Choose a context and slot count that fit
on every rank.

Use `packed.open_packed_session` with a matched graph pair, the resident
checkpoint weights and HostEngram. One worker thread owns the session.
The caller chooses the phase, DP owners and request set for each dispatch.

```python
a = session.admit("a", dp=0, prompt_length=3)
b = session.admit("b", dp=0, prompt_length=2)
first = session.advance_batch("prefill", [(a, [11, 12]), (b, [21, 22])])
tail = session.advance_batch("prefill", [(a, [13])])
next_rows = session.advance_batch("decode", [
    (b, [int(first[b].argmax())]), (a, [int(tail[a].argmax())]),
])
session.release(a)
session.release(b)
```

Outputs are full-vocabulary logits keyed by the original session-local handle.
An intermediate prompt chunk yields no logits for that request. Omitted
requests retain their pages and history. Release a finished or cancelled
request after its dispatch returns. Reusing its ID creates a new handle.
After a device or reset failure, close the session.
