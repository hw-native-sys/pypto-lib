# Engram device lookup and forward inputs

Engram tables use device-resident FP8 E4M3 payloads and native group-32 E8M0
scales. `engram.py` owns token compression, hashing, lookup, communication, WKV and gate. Both forwards prepare both backbone Engram layers, 1 and 14,
before the backbone executes. Each requesting DP rank receives BF16 rows;
WKV projection and gate computation execute on that requesting rank. There
is no CPU offload or host lookup fallback in this path.

## Sharded lookup

`--ep` selects the DP communication group and `--engram-shards` selects its
first table-owning ranks. Original row `id` belongs to `id % shards`. A full
shard stores that row at `id // shards`. These are independent DP requests:
rank-local token sequences and counts can differ, including zero tokens.
An empty requesting rank still serves incoming requests for its table shard.

The lookup program uses two exchanges:

1. Request ranks pack original hash IDs in layer/token/head order and issue
   peer puts into each owner's communication window. Per-source completion
   counters cover every request block, including empty blocks.
2. Owners wait for requests, load native payloads and scales, dequantize on
   the NPU and peer-put BF16 rows and missing-row status back to each source.
   Sources wait for all owners and restore original layer/token/head order.

Each request slot has one owner and one response writer. Masked and inactive
rows are explicitly zeroed. A reader-completion handshake precedes signal
retirement. Calls must be serialized across the entire DP group: the previous
`runtime.run` must complete on every rank before a new call reuses its
windows. Every rank must participate, even when all ranks have zero tokens.
The host helper checks the returned status before accepting logits or committing cache state.

The request capacity is compiled; resident table and compact-map row counts
are dynamic. The BF16 output capacity is also dynamic and may be smaller
than the lookup capacity, allowing one lookup artifact to fill both prefill
and decode buffers. Counts must fit both capacities. This implementation
uses bounded padded request slots rather than a TP all-reduce. It returns
BF16 data; FP8-plus-scale response compression is future work.

## Rank-local forward ABI

Both forward L2 entries execute the complete device Engram preparation. Let `T` be prefill's
physical `CAPACITY` or decode's `LOCAL_T_DYN`. L3 adds a leading DP rank axis.

| Input | Rank-local shape | Storage |
| --- | --- | --- |
| `engram_context` | `[T, 4]` | INT32 raw current/history IDs, -1 for excluded context |
| `engram_token_map` | `[vocab_size]` | INT32 tokenizer compression LUT |
| `engram_multipliers` | `[2, 4]` | INT64 layer-specific hash multipliers |
| `engram_primes`, `engram_offsets` | `[2, 24]` | INT32 bucket sizes and head offsets |
| `engram_table`, `engram_scale` | `[2, resident_rows, 256]`, `[2, resident_rows, 8]` | Native FP8 / E8M0 owner shards |
| `engram_row_ids` | `[2, map_rows]` | INT32 sorted original-ID map in compact mode |
| `engram_rows` | `[2, T, 6144]` | BF16 lookup output, written by the forward |
| `engram_hash_ids`, `engram_missing` | Compiled request capacity | Hash/status outputs for checking |
| `engram_mask` | `[T]` | INT32 text keep mask |
| `engram_wkv` | `[2, 6144, 25600]` | FP8 E4M3, K-N order |
| `engram_wkv_scale` | `[2, 192, 25600]` | E8M0, MX B-NN order |
| `engram_gate_weight` | `[2, 4, 5120]` | FP32 product of checkpoint Q/K gate weights |

A token's 6144 values concatenate 24 heads, with 256 channels each, in n-gram
order (2, 3, 4), then head order. The device lookup fills the forward's own
`engram_rows` allocation. The forward preserves `engram_rows → WKV → gate`,
injecting the result before each Engram layer's attention mixes and collapse.
The incoming delayed pre-mix remains unchanged. Masked and inactive hidden
rows pass through exactly; active updates preserve BF16 projection and
residual rounding boundaries inside the FP32 residual stream.

`EngramDeviceTables` streams full native shards during initialization or
constructs compact native shards from an explicit collection of hashes.
The L3 forwards allocate communication windows and dispatch one complete L2
forward per rank. Both `l2_prefill_fwd` and `l2_decode_fwd` receive raw contexts,
native tables and communication-window views. They inline the shared
`prepare_engram` computation before the backbone: token compression, both
layers' hashes, request/response exchange and BF16 row restoration. There is
no separate lookup submission. The standalone lookup entry wraps the same
inline computation for independent protocol validation.

The request count is rank-local: prefill supplies `engram_counts` for its
encoder rows; decode reuses `attention_num_tokens`. `engram_mask` follows
hidden-row order. `engram_context[t, j]` is the raw token at the current token's
absolute position minus `j`. A missing negative-position token or excluded
token is `-1`; contexts crossing it remain padded. The serving caller supplies
correct request history. Token compression and hash arithmetic execute on
the requesting NPU. The returned missing status must be checked before
accepting logits or committing cache state. A nonzero status is a failed
forward, requiring cache restoration or reset before retry.

`prepare_prefill_inputs` and `prepare_resident_inputs` retain BF16 placeholder
rows for metadata allocation. Production generation skips their upload:
`engram_rows` is an output filled inside the forward. Previously compiled
row-input-only artifacts must be rebuilt. The artifact sidecar checks rank,
shard, request-capacity and compact-mode specialization against loaded tables.

## Hashing and token history

`engram.EngramLookup.prepare_context` packs raw token history without hashing
or fetching rows. Its `hash_ids` method computes CPU reference hashes. Tokenizer compression, layer-specific hash multipliers, prime buckets
and head offsets follow the checkpoint layout. The helper validates the
compressed vocabulary size. Its separate `prepare_rows` method remains a
host reference for validation and existing gathered-row callers.

Packed metadata includes raw token IDs, absolute positions, query starts and
`lookback_token_ids[requests, 3]`. Column `j` contains the token at
`chunk_start - 1 - j`; negative historical positions may use `-1`. Positions
inside each request chunk must be consecutive. Zero masks exclude tokens
and break n-grams across them; historical masks use the same semantics.

Serving owns request scheduling, chunk transitions, prefix-cache hits,
prefill/decode migration and speculative rollback. Preserve or reconstruct
the last three raw input IDs and masks; KV state alone is insufficient.
Commit history only after successful forward execution. Both Engram layers
depend only on these token inputs and can be prepared together before the
backbone.

## Compact real-table validation

A compact manifest is a CPU list of hash tensors ending in `[2, 24]`, saved
with `torch.save`. Collect hashes in advance for each complete test prompt
and its fixed continuation, including the correct prompt encoding and token
history. Each owner deduplicates and sorts original IDs, copies their native
FP8/E8M0 bytes from the checkpoint and stores an explicit original-ID map.
The compact kernel searches that map; it never treats a compact index as an
original hash ID.

The host rejects IDs outside the collected set before dispatch. The device
also reports unmapped IDs, out-of-range rows and negative active IDs. A
free-generation miss is an error: extend the manifest and rerun, or stop.
No row substitution, zero-result acceptance or host lookup fallback is allowed.

## Capacity and validation scope

All values below are decimal GB, assuming 96 GB per card.

| EP8 component | Weight GB/card |
| --- | ---: |
| Main model, including Engram WKV/gates | 45.0163 |
| Native Engram tables, eight shards | 25.3448 |
| Replicated ViT and aligner | 0.9705 |
| Drafter, routed experts sharded by EP8, other weights replicated | 1.6159 |
| Complete replicated embedding | 1.3238 |
| Total | 74.2713 |

The table excludes KV cache, activations, runtime/communication buffers and
allocator fragmentation. Uniform table-shape padding adds less than 0.001
GB/card. The approximately 21.73 GB/card remainder is a static budget, not a
measured usable-memory guarantee. Full-table EP4 requires about 81.11 GB of
main weights plus 50.69 GB of tables per card and exceeds capacity. The
four-card generation example therefore requires a compact manifest.

Compile compact four-rank lookup and full eight-rank lookup without devices:

```bash
python models/deepseek_v4_1_flash/engram.py --validate-lookup --ep 4 --engram-shards 4 --lookup-capacity 128 --engram-compact --compile-only
python models/deepseek_v4_1_flash/engram.py --validate-lookup --ep 8 --engram-shards 8 --lookup-capacity 128 --compile-only
python models/deepseek_v4_1_flash/prefill_fwd.py --ep 8 --capacity 128 --compile-only
python models/deepseek_v4_1_flash/decode_fwd.py --ep 8 --compile-only
```

Run four-card independent synthetic-table validation through the local allocator:

```bash
task-submit --device auto --device-num 4 --max-time 600 'PYTHONPATH=. python models/deepseek_v4_1_flash/engram.py --validate-lookup --ep 4 --engram-shards 4 --lookup-capacity 4 --engram-compact -d "$TASK_DEVICE"'
```

The entry checks every BF16 element and status over six retained-window calls,
including local/remote hits, different requests, empty owners, an all-empty
call, masks, negative IDs, compact misses and recovery. Repeat with
`--engram-shards 2` for non-owner ranks, or omit `--engram-compact` to validate
the full-table striped indexing on a small synthetic table.

Both forward CI entries retain `# ci: devices=2` and their existing fixed
expert count. CI supplies an explicit small hash namespace (24 buckets of
size 17 per layer), native FP8/E8M0 shards and independent raw token contexts.
The kernel performs the same device hash, peer exchange, dequantization, WKV
and gate path. CI checks all generated hash IDs and BF16 rows against the CPU
reference after the forward. This bounds table memory without changing the
production bucket configuration or requiring the complete checkpoint.
With 64 experts per card, main test weights occupy about 56.84 GB/card,
including the full-size Engram WKV/gates. The bounded native tables add
107,712 bytes/card, plus 1,632 bytes for the original-ID map and 517,120 bytes
for the tokenizer compression LUT.

CI's bounded synthetic-table checks are separate from compact real-table
validation, which retains the production hash configuration and original IDs.

Four-card compact real-table validation and complete EP8 residency are distinct
acceptance levels. EP8 compilation and static weight accounting do not establish
full-table loading, measured peak memory, performance or end-to-end accuracy;
those require a later eight-card device run.

Validation on four cards passed six independent lookup calls, including
empty ranks and buffer reuse, and compact real-checkpoint generation with
one complete prefill and eleven decode calls. Device hashes and every BF16
lookup element matched the CPU oracle exactly; all 129,280 logits at each
active request step matched the earlier host-lookup forward exactly. All four
fixed continuations reached the expected EOS. Default two-card prefill/decode
CI runs also passed exact hash/row checks. Full-table EP8 prefill/decode
compilation passed; complete EP8 residency remains unmeasured.

References: [vLLM Engram](https://github.com/vllm-project/vllm/blob/c2861f7f120d508dcb1be046d9fc659c93062508/vllm/models/deepseek_v41/common/engram.py)
and [model composition](https://github.com/vllm-project/vllm/blob/c2861f7f120d508dcb1be046d9fc659c93062508/vllm/models/deepseek_v41/nvidia/model.py).
