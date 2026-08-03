# Qwen3

PyPTO-Lib tracks Qwen3 kernels for 14B and 32B variants. The 14B tree contains
the serving-facing prefill/decode contract and several component harnesses. The
32B tree currently contains single-layer decode experiments.

## Qwen3-14B

### Runnable and imported entries

| Entry | Classification | Declared platforms | Configured CI coverage |
| --- | --- | --- | --- |
| [decode_fwd.py](../../models/qwen3_14b/decode_fwd.py) | Full decode implementation; the CLI defaults to a single-layer golden and uses `--validate-fwd` for a stacked forward | A2/A3, A2/A3 sim | A2/A3 device; marked `no-sim` in CI |
| [prefill_fwd.py](../../models/qwen3_14b/prefill_fwd.py) | Multi-layer BF16 prefill forward | A2/A3, A5 | A2/A3 device; marked `no-sim` in CI |
| [decode_layer_a8w8.py](../../models/qwen3_14b/decode_layer_a8w8.py) | One-layer A8W8 smoke harness | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [greedy_sample.py](../../models/qwen3_14b/greedy_sample.py) and [topk_select.py](../../models/qwen3_14b/topk_select.py) | Sampling component harnesses | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [test_paged_attention_cce.py](../../models/qwen3_14b/test_paged_attention_cce.py) | On-device CANN paged-attention correctness/performance driver | A2/A3, A2/A3 sim | A2/A3 device; marked `no-sim` in CI |
| [rope_qkv_regen.py](../../models/qwen3_14b/rope_qkv_regen.py) | Compile-only regeneration source for the embedded CCE header | A2/A3, A2/A3 sim | A2/A3 compile sweep; not a runtime correctness case |
| [prefill_fwd_a8w8.py](../../models/qwen3_14b/prefill_fwd_a8w8.py) | Imported A8W8 prefill implementation, no standalone CLI | None | No direct per-file run |

The BF16 decode path calls a CANN
`FusedInferAttentionScore` bridge and is explicitly A2/A3-oriented. The
simulator option on `decode_fwd.py` compiles a smoke program, but the file's
`no-sim` marker means the daily simulator sweep skips it.

The 14B [contract module](../../models/qwen3_14b/contract.py) registers BF16
prefill and decode stages for external runtimes. Changes under the 14B tree
also gate a one-card A2/A3 `pypto-serving` accuracy test in pull requests. That
is a real-weight integration path; the direct CLIs normally build synthetic
fixtures.

### Draft and utility code

The following files are drafts and are excluded from CI:

- [decode_ssn_draft.py](../../models/qwen3_14b/decode_ssn_draft.py):
  serial, 4D-blocked single-layer decode.
- [decode_tq_draft.py](../../models/qwen3_14b/decode_tq_draft.py):
  TurboQuant decode.
- [prefill_tq_draft.py](../../models/qwen3_14b/prefill_tq_draft.py):
  TurboQuant prefill.

Configuration, weights, final RMS/LM-head, TurboQuant operators, and the CANN
bridge are library modules. They have no standalone CI entry simply because
they have no `__main__` block; importing harnesses provide their coverage.

### Run the 14B harnesses

Run the default decode golden on one A2/A3 device:

```bash
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0
```

Validate a stacked decode forward instead of the default single-layer case:

```bash
python models/qwen3_14b/decode_fwd.py \
  -p a2a3 -d 0 --validate-fwd --fwd-layers 4
```

Run a two-layer synthetic prefill fixture:

```bash
python models/qwen3_14b/prefill_fwd.py \
  -p a2a3 -d 0 --num-layers 2
```

These cases can allocate large fixtures. Use the script-specific `--help`
before increasing batch, sequence length, or layer count.

## Qwen3-32B

| Entry | Classification | Declared platforms | Configured CI coverage |
| --- | --- | --- | --- |
| [decode.py](../../models/qwen3_32b/decode.py) | Single-layer decode, conventional tensor layout | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [decode_4d.py](../../models/qwen3_32b/decode_4d.py) | Single-layer decode, 4D-blocked layout | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| [prefill_draft.py](../../models/qwen3_32b/prefill_draft.py) | Draft single-layer prefill | A2/A3, A5 | Excluded because it is a draft |

The 32B entries are component harnesses, not serving contracts or full
multi-layer runners. A simulator run is a suitable first check:

```bash
python models/qwen3_32b/decode.py -p a2a3sim
```
