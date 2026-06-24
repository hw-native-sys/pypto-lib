# Qwen3-14B Prefill — Ring / Arena Sizing & Long-Context Limits

Findings from sizing the PTO2 ring resources for
[`prefill_fwd.py`](prefill_fwd.py) on **a2a3 (910B, 64GB HBM)**, derived from
on-device experiments with `--enable-scope-stats`.

## TL;DR

- The runtime's **pooled static arena scales with `PTO2_RING_TASK_WINDOW`**:
  `arena_bytes ≈ 19,614 × PTO2_RING_TASK_WINDOW` (HEAP / DEP_POOL negligible).
- Single-shot prefill `task_window` peak grows with sequence length, so the arena
  blows past HBM above **~1300–1400 tokens**.
- **Long contexts (e.g. 3500) must use chunked prefill** (`--chunk-start` /
  `--chunk-size`); historical KV is cached, so per-chunk `task_window` stays small.

## How to diagnose

Enable per-scope ring-fill capture (flag wired into `runtime_cfg`):

```bash
python models/qwen3/14b/prefill_fwd.py -p a2a3 -d 0 --max-seq 128 --enable-scope-stats
```

Peaks land in `build_output/<run>/dfx_outputs/scope_stats/scope_stats.jsonl`.
Each record carries `task_window_end` / `dep_pool_end` / `heap_end` / `tensormap`
per `ring` / `depth` / `site`. The first line is a header with the configured caps.

> **Caveat:** peaks from a *crashed* run are truncated — the device drains early,
> so the recorded peak under-reports true demand. Always re-measure on a passing run.

## The two failure modes

### 1. Ring overflow → AICore error 507018

Caps too small for the workload. The per-token attention scope at
`prefill_fwd.cpp:172` (ring3) overflows `task_window` / `dep_pool`:

```
sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
AICore error 507018: device drained
```

Fix: raise `PTO2_RING_TASK_WINDOW` / `PTO2_RING_DEP_POOL` to just above the
measured peak.

### 2. Arena OOM at prepare → rtMalloc 207001 / code -1

Caps too *large* — the static arena exceeds HBM at bind time:

```
rtMalloc failed: 207001 (size=164550936383)   # ~153 GB requested
Failed to setup pooled static arena            # runtime_maker.cpp:357
RuntimeError: run_prepared failed with code -1
```

This happens **before** runtime execution, so no `scope_stats` is written.
Fix: *lower* `PTO2_RING_TASK_WINDOW` toward the true peak — do not over-provision.

## The arena formula

| `PTO2_RING_TASK_WINDOW` | arena requested | result |
| --- | --- | --- |
| 8,388,608 | 164.5 GB | OOM |
| 4,194,304 | 82.3 GB | OOM |
| 1,048,576 | 20.6 GB | OK (fits 64GB card w/ 28GB weights) |

`arena ≈ 19,614 × TASK_WINDOW`. HEAP and DEP_POOL do not materially change it.

> The `heap` dimension in scope_stats always reads ~99.97% of `PTO2_RING_HEAP`
> regardless of the configured value (4GB run and 8GB run both ~100%) — it is the
> allocator's config-proportional arena reservation, **not** real heap pressure.
> Don't chase it; it is not the bottleneck.

## Measured per-ring peaks (ring3 dominates)

| seq | task_window | dep_pool | scaling vs prev |
| --- | --- | --- | --- |
| 128 (single-shot) | 205,824 | 251,775 | — |
| 512 (single-shot) | 617,472 | 755,292 | ~3.0× for 4× seq |
| 3500 (single-shot) | ~3.8M (predicted) | ~4.7M (predicted) | arena ~70GB → **infeasible** |
| 3500, chunk-size 512 @ start 2988 | **68,608** | **84,164** | fits with caps ≥ 131072 (built-in default is lower → still overflows) |

## Recommended configs

| Scenario | TASK_WINDOW | DEP_POOL | HEAP | Status |
| --- | --- | --- | --- | --- |
| single-shot, seq 128 | 262144 | 524288 | 4 GB (4294967296) | verified PASS |
| single-shot, seq 512 | 1048576 | 1048576 | 8 GB (8589934592) | verified PASS |
| **seq 3500 (and any long ctx)** | **chunked, 1048576** | 1048576 | 4 GB | verified PASS; single-shot infeasible |

### Long context: chunked prefill recipe

Single-shot prefill is capped at ~1300–1400 tokens by the arena. For 3500, iterate
512-token chunks; per-chunk `task_window` peak ≈ 69k, `dep_pool` ≈ 84k — small, but
still above the built-in default ring caps, so you must still pass `--env` overrides
(just modest ones; arena at TASK_WINDOW=1048576 ≈ 20.6 GB):

```bash
# Heaviest (last) chunk of a 3500-token context — verified PASS, peak tw 68,608 / dp 84,164
PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=1048576 PTO2_RING_DEP_POOL=1048576 \
  python models/qwen3/14b/prefill_fwd.py -p a2a3 -d 0 \
    --max-seq 3500 --chunk-start 2988 --chunk-size 512

# Full prompt: sweep chunk-start = 0, 512, 1024, ..., 2988 (seven 512-chunks)
```

> Caps ≥ 131072 are enough for the measured peaks; the *built-in* default (env unset)
> is lower than the ~84k dep_pool peak and overflows with 507018, so overrides are
> still required — just far smaller than single-shot would need.
