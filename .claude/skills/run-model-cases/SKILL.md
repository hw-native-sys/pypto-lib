---
name: run-model-cases
description: Pick and run the right model case in this repository — which entry point validates a single layer versus a multi-layer forward, which platform and how many cards each one needs, and what its passing output actually looks like. Use when someone asks to run a model, to validate a layer or forward path, or needs the cheapest case that exercises a tree.
---

# Run a Model Case

This skill owns the case inventory: which entry point to run, on what
platform, with how many cards, and what its pass signal is. It is the answer to
"run Qwen decode" or "validate the DeepSeek layer", not a substitute for the
model documentation, which explains what each program computes:

- [Models](../../../docs/models/index.md)
- [Qwen3-14B](../../../docs/models/qwen3_14b/index.md)
- [DeepSeek V4-Flash](../../../docs/models/deepseek_v4_flash_mtp/index.md)
- [Platforms and Devices](../../../docs/get-started/platforms.md)
- [Compile and Runtime Workflow](../../../docs/run-and-validate/compile-runtime-workflow.md)

For a caller who has not yet proven the environment, the setup sequence and its
gates live in the `setup-and-run` skill. Do not re-derive them here; this skill
assumes an environment that already runs a case.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Before running anything

Entry-point names, flags, and platform support change with the tree. Confirm
against the file, not against memory or this table:

```bash
PYTHONPATH="$PWD" python models/<family>/<entry>.py --help
grep -n '^#\s*ci:' models/<family>/<entry>.py
```

Read the markers as hard constraints. `# ci: no-sim` means the case is
device-only. `# ci: devices=N` means it needs N allocated cards. A platform a
parser accepts is not a platform the case is validated on.

Run every case from the repository root with that root on `PYTHONPATH`.

## Qwen3-14B — single card

| Goal | Command |
|---|---|
| One decode layer | `python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0` |
| N stacked decode layers plus the LM head | `python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0 --validate-fwd --fwd-layers 4` |
| Multi-layer prefill | `python models/qwen3_14b/prefill_fwd.py -p a2a3 -d 0 --num-layers 2` |

`decode_fwd.py` serves both decode goals from one file. Without
`--validate-fwd` it runs the single-layer golden and `--fwd-layers` is ignored
entirely; with it, the run switches to the stacked forward validated against a
host chain reference. A layer count that appears to have no effect is almost
always this flag pair.

Both entry points are marked `no-sim`. The decode path reaches attention
through a vendor CCE bridge that is A2/A3 onboard only, so this tree has no
cheap simulator rung — a broken toolchain surfaces here as a device exception
rather than as a compile error.

The two goals report differently, and only one of them prints the harness line:

- the single-layer case ends in the harness `[RUN] PASS` after an `'out' PASS`
  tensor comparison;
- the stacked forward prints its own summary instead —
  `[stacked-fwd 4L+LMhead] ... argmax match 16/16 | sample match 16/16 |
  logits ...% within 5e-2 | max_abs_err=...` — and never prints `[RUN] PASS`.

Quote whichever line the case actually emitted. Do not report a stacked-forward
run as passing on exit status alone.

## DeepSeek V4-Flash — two cards by default

| Goal | Command |
|---|---|
| One attention path | `python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0` |
| One full layer | `python models/deepseek_v4_flash_mtp/decode_layer.py -p a2a3 --ep 2 -d 0,1` |
| Multi-layer forward | `python models/deepseek_v4_flash_mtp/decode_fwd.py -p a2a3 --ep 2 -d 0,1` |

`-d` here is a comma-separated device set, not an integer, and `--ep` is the
rank count that the MoE module parses at import. The layer and forward cases
carry `# ci: devices=2`; their per-file CI coverage is the default two-rank
case only, so a larger `--ep` is an experiment rather than a covered path.

Component entry points such as `decode_csa.py` and `rmsnorm.py` take a single
card. Their CLIs also accept a simulator, but a simulator run can fail in the
runtime stage on workspace provisioning while the same case validates on a
device — prefer the device form when a card is available.

DeepSeek V4 Pro mirrors this tree on A5 (`-p a5`).

## Escalating within a tree

Start at the case that answers the question. Fall back to a cheaper one only
when it fails, and then only to localize: a component case that passes while a
layer case fails separates a kernel fault from a composition fault, and a
layer case that passes while a forward fails separates a layer from the
schedule around it. Do not make a caller climb a ladder they did not ask for.

Allocate devices through the site's allocator rather than by probing for an
idle card, and pass the allocated ids.

## Cost discipline

Forward cases allocate large fixtures and can exhaust host memory or the device
cache pool. Keep batch, sequence length, layer count, and world size at their
defaults on a first run, then raise one at a time, reading `--help` first.

For repeated runs whose numerics are unchanged, freeze the reference once with
the `test-with-golden` skill instead of recomputing the torch golden every
iteration.

## Reporting

Report the exact command, the platform and device set, and the pass line the
run printed. A validation mismatch is a numerical failure and belongs in
[precision tuning](../../../docs/debug-and-tune/precision-tuning.md); a hang or
an accelerator error code belongs in
[debugging](../../../docs/debug-and-tune/debugging.md) — but on an environment
that was not just proven, re-check the toolchain pins before either, because a
skewed assembler produces both.
