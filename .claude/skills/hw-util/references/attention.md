# Forward attention theoretical MFU

Use for forward FlashAttention (FA), including GQA, MQA, MLA and sparse
attention. Calculate MFU only; FA MBU and bandwidth estimates are deferred.
Use `scripts/calculate_fa.py` from the skill directory. The GEMM calculator's
read/write estimates do not describe a fused FA invocation.

## Theoretical work and timing

Count the mathematically required QK and PV contractions in the selected FA
formulation, using semantic dimensions and semantic query-key pairs. Hardware
requirements never increase this numerator: exclude tile rounding, padded
sequence/head dimensions, masked dense pairs, sparse capacity padding,
sentinel entries, redundant tile work and hardware-induced recomputation.
Derive counts from the model inputs, mask and sparse-selection contract before
examining generated tile loops. Those loops describe executed work, not the
theoretical numerator.

Let `E` be the total pair count across batch, query heads and query positions.
Let `D_QK` and `D_V` be the unpadded contraction and value dimensions:

```text
F_QK = 2 * E * D_QK
F_PV = 2 * E * D_V
F_theoretical = F_QK + F_PV
effective_TFLOPS = F_theoretical / (t_us * 1e6)
MFU_pct = 100 * effective_TFLOPS / peak_TFLOPS
```

Use the full FA invocation span for `t_us`, following the skill's timing rules.
Softmax, masking, rescaling and hardware-added work consume time but do not add
Cube FLOPs to this numerator. This is theoretical useful-work normalization
over observed time, not a prediction of latency. A mean-core hypothesis must
retain its hypothetical timing label.

Match the device Cube peak to the actual compute mode. The helper assumes the
same peak applies to both QK and PV. If their compute modes have different
peaks, report that limitation and derive a separate, explicit per-stage model;
do not silently apply one peak to both. FA sparsity changes `E`; it does not
by itself justify a hardware structured-sparse peak.

GQA/MQA use the number of **query heads** for compute, even when KV heads are
shared. For MLA, establish the semantic dimensions of the selected formulation,
including latent dimensions after projection absorption when applicable.
Exclude alignment padding and projections outside the named FA scope. The
helper permits different `D_QK` and `D_V`; it does not infer them from a model
name or assume they are equal.

## Ordinary attention: count mask-valid pairs

For GQA/MQA/MLA without sparse selection:

```text
E = sum over batch, query heads and queries of number_of_valid_unmasked_keys
```

- `--pair-mode dense`: no mask; `E = B * Hq * Lq * Lkv`.
- `--pair-mode causal`: include the diagonal. Equal lengths `S` give exactly
  `E = B * Hq * S * (S + 1) / 2`, rather than halving `S*S`.
- `--pair-mode masked`: provide `--effective-pairs E`, already summed across
  batch, query heads and queries. Use this for arbitrary masks, windows,
  ragged lengths or combinations of padding and causal masks. Do not multiply
  this aggregate by batch or heads again. Zero valid pairs is supported.

For unequal query/KV lengths, causal alignment must be explicit. With a
zero-based query position `i`:

```text
top-left:     valid_keys(i) = clamp(i + 1, 0, Lkv)
bottom-right: valid_keys(i) = clamp(i + 1 + Lkv - Lq, 0, Lkv)
```

The helper sums these counts exactly. A single bottom-right-aligned decode
query sees the complete KV prefix; do not halve its work. Additional masks
require the explicit aggregate mode rather than a generic causal factor.

## Sparse attention: count every semantic selected entry

Count every entry required by the sparse-selection contract, without applying
an additional causal or triangular discount:

```text
E_sparse = sum over batch, query heads and queries of selected_entry_count
uniform selection: E_sparse = B * Hq * Lq * K_selected
```

Use `--pair-mode sparse --selected-k K_selected` for a uniform count, or
`--pair-mode sparse --effective-pairs E_sparse` for variable counts.
`K_selected` is the semantic selection size, not a rounded buffer capacity.
For example, 70 selected entries stored in 128 slots contribute 70 entries;
the extra 58 slots and hardware-repeated calculations contribute no FLOPs.
An unknown selection count cannot be replaced by capacity or a guessed fraction.

For ragged inputs, explicit totals must use the true per-query selections.
The CLI dimensions bound the input domain; they do not add padding work to an
explicit total. The caller must verify that the total excludes padding.

## Examples and output

Illustrative causal arithmetic on a 32-AIC 950DT BF16 compute path, with a
hypothetical 100-us full invocation span:

```bash
python .claude/skills/hw-util/scripts/calculate_fa.py \
  --batch 1 --query-heads 8 --q-len 128 --kv-len 128 \
  --qk-dim 64 --v-dim 64 --pair-mode causal \
  --time-us 100 --time-kind kernel-span --peak-tflops 432
```

This has 66,048 semantic pairs and 16,908,288 theoretical FLOPs. Masked
positions inside executed square tiles remain excluded.

For a sparse formulation with 70 selected entries per query and unequal
semantic QK/value dimensions:

```bash
python .claude/skills/hw-util/scripts/calculate_fa.py \
  --batch 1 --query-heads 8 --q-len 1 --kv-len 8192 \
  --qk-dim 576 --v-dim 512 --pair-mode sparse --selected-k 70 \
  --time-us 100 --time-kind kernel-span --peak-tflops 432
```

This has 560 semantic pairs and 1,218,560 theoretical FLOPs, regardless of
whether execution rounds selection storage to 128 entries. Both examples are
arithmetic demonstrations, not performance measurements.

Report the pair-count evidence, mask/alignment or sparse selection, unpadded
dimensions, QK/PV FLOPs, full timing window, compute dtype and peak source,
effective TFLOPS and MFU. Omit MBU and bandwidth fields from FA JSON and reports,
including null values and not-computed placeholders. Do not request HBM traffic
or derive FA MBU as part of this workflow.
