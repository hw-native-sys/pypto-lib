# F02 worker: original-source rotary/output oracle

Delivered artifact-only candidate-independent oracle. No model, golden,
threshold, ledger, checkpoint or PyPTO files were written. No NPU jobs or
subagents were launched. This is Main-owned worker self-evidence, not independent
acceptance. Worker parks after this delivery.

API in oracle.py:

- build_fixture(tokens) -> attended[T,8192] BF16, cos/sin[T,32] FP32,
  wo_a[2,1024,4096] BF16 and wo_b[5120,2048] BF16.
- expected_output(tensors, mode='inverse') -> BF16[T,5120], with additional
  modes missing and forward for controls. This uses paired complex arithmetic,
  explicit BF16 rounding and a sparse three-term/two-term projection closed form.
- original_rows/original_output execute AST-extracted preserved rc2 adapter,
  rotary CPU golden, _wo_a_bmm and _forward_o_proj with CPU primitive stand-ins.
- provenance verifies the preserved archive hash and exact bytes of all four
  reference sources, whose individual hashes are in host_results.json.

No candidate kernel, projection, rotary, attention, golden or config is imported.
Angles vary by token and rotary pair, with both signed rotary/nonrotary values,
distinct head/group channels and asymmetric Wo coefficients. Every Wo_a output
has exactly three terms (adjacent rotary pair plus a nonrotary lane); every
Wo_b output has two distinct terms from the grouped BF16 intermediate.

## Reference grounding

Official rc2 revision 8fde6d4e335fe300e2b949aab0e31659cce0811d, archive
SHA25648c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c.
Preserved source root: logs/logic_review_1793929657/upstream_rc2.

- dsa_v41.py:711–716 applies inverse query-position rotary to attention output
  before _project_output. Its _apply_rotary at291 passes start/end/inverse.
- device/device_op.py:76–90 performs rank adaptation and chooses negate_sin
  capability or explicitly negated sine. The executed stand-in selects the
  explicit -sin path, equivalent to inverse adjacent-pair complex rotation.
- Original test_inplace_partial_rotary_mul.py:16–44 uses FP32 rotary arithmetic,
  adjacent even/odd partners and unchanged nonrotary prefix. It returns FP32;
  the inplace operator destination is BF16 here, so the CPU stand-in explicitly
  rounds on copy into that BF16 destination. Original test defaults use a192-lane
  head with128-lane prefix; AST execution binds NOPE_DIM=448 for the same64-lane
  rotary tail of this512-lane model.
- dsa_v1.py:1732–1757 _wo_a_bmm uses non-FP8
  npu_transpose_batchmatmul on BF16 operands. _forward_o_proj:1759–1840 reshapes
  into local groups, obtains the BMM intermediate, flattens and calls Wo_b.
  The non-OTP branch is AST-executed without rewriting its statements. CPU
  transpose-BMM and linear stand-ins accumulate FP32 and round their BF16
  outputs, preserving the existing candidate's selected grouped BF16 contract.

This does not execute original torch_npu primitives on hardware or original
full model/checkpoint. It does not establish a primitive's exact reduction
arithmetic, tie quantization parity, or full TP/DP collective semantics.

## Measured CPU check

Queue task task_20261106_114552_1950517298, --no-device, exit0. No NPU allocated.
Exact command from repository root (recorded in host_queue.log):

```bash
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && export PATH="$PTOAS_ROOT/bin:$PATH" && python experiments/dsv41-flash-w8a8-a3/logs/attention_f02_1793936494/independent_oracle/oracle.py'
```

All T1/T7/T17 closed-form expected outputs exactly equal the original-source
stand-in outputs. Nonrotary lanes exactly retain their input BF16 values.
Both control outputs independently match their original-source counterparts.
At the unchanged output element gate |err| > .02+.02*|expected|:

| Tokens | Missing inverse rejected lanes | Max abs | Forward rejected lanes | Max abs |
| --- | --- | --- | --- | --- |
| 1 |4327/5120|1.28125|4827/5120|2.18359375|
| 7 |32497/35840|1.84765625|34471/35840|2.5546875|
| 17 |81523/87040|2.53125|79691/87040|2.5546875|

The script rejects any control with <=0.5% failing lanes and propagates
exceptions. These are CPU semantic controls; Main must still demonstrate
compiled candidate/device numerical rejection. Main alone owns integrating
permanent model defaults and device runner evidence.

## Recovery

oracle.py SHA256567dee516d565aaa8f624d14bd41298ab83cedcb9122ef0394ef66d6a5bab82f.
host_results.json contains reference hashes/results, host_queue.log contains
exact submitted task identity and exit. Worker task complete, no live jobs or
sessions remain. Await Main request before any further edits. F03/F04 and all
other open/held gates are unaffected.
