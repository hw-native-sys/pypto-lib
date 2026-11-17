# Independent HC draft validation and prepared native microproof

**CPU/reference and signature-compile review PASS. Native PREPARED ONLY; NOT EXECUTED.** Repetition causality remains UNPROVEN. No live math, generation, runtime, dependency or prior artifact changes.

## Frozen candidate and review

Reviewed Main's unapplied `hc_bf16_boundary_draft_1794071285`: manifest SHA256 `5a2ccfb048c16742248c158708711d8ac6bd7e3ae586528336d5065b1d02586f`; patch `7520f4c6d7f144b0a2937751b9f43ad6c45c653c4b8fca49606c8d0dc4134c72`. Proposed HC SHA256 `c6111634f78d53ad0cb4196a533a2d89097d4cfa0859688382c0072b60840885`, proposed layer-reference `c382bc7131c4bfd3f159d291c99de7e0d9f3368e5b25e1c3515b96f825dedda0`. Byte-identical owned snapshots are in `sources/`; `source_pins.json` binds their original paths. Live source45 manifest `cc15877f8f4b0104a842f915d2434d9a2c9cd6f6752d6364d5d7bada26e21d8c` was reverified unchanged (45/45).

Four direct code anchors establish the boundary:

1. `sources/proposed_hc.py:214-216`: final accumulated FP32 value → BF16 `mode="rint"` → FP32 store.
2. `sources/proposed_hc.py:330-334`: actual proposed standalone golden rounds its final sum to BF16 and promotes back to FP32.
3. `sources/proposed_layer_reference.py:155`: actual composed golden returns `(update + skip).to(torch.bfloat16).float()`.
4. `sources/original_utils.py:266-270`: actual pinned original `hc_post_reference` returns `y.to(x.dtype)`; BF16 sublayer input therefore determines the boundary.

The public FP32 residual/output ABI is preserved. Independent AST comparison confirms only `hc_post` and `golden_hc_post` change in HC; HC pre/head/wrapper signatures and other functions are identical. The two-file diff contains no extra arithmetic/caller/tolerance changes. Original rc2 archive SHA256 is `48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`, commit-prefix `8fde6d4e335fe300e2b949aab0e31659cce0811d`; original helper file SHA256 `577c667fbf8f8b6f20de8a0ac54c6c0435605aaee1f6c2ccc463b446055bd286`. R01's earlier byte-verification supplies the proto/A3-kernel grounding; no new original is invented.

No approved FP32-residual exception was located in goal/MODEL_SPEC/MODULES. The previously considered routing FP32 issue remains rejected: original RMSNorm-cast explicitly preserves rounded routing input.

## Independent queued CPU proof

Task `task_20261108_013235_332385828347`, **exit0**, used the existing activated environment, `--no-device`, Torch `2.14.1+cpu`. `probe.py` executes actual original/proposed golden function bodies unchanged via AST extraction. Original residual is BF16; proposed FP32 storage contains the identical BF16-exact values. Coefficients are FP32.

Five rows cover signed reproducer, signed even/odd ties, either side of a midpoint, nonidentity dyadic residual mixing, and already-exact output. Both proposed goldens match original promoted output **bit-for-bit: zero mismatches in 102400 elements**. Every candidate FP32 result is exactly BF16-representable. Old omitted-boundary per-row mismatch/nonrepresentability counts are **15360, 20480, 20480, 5120, 0**; max error 0.00390625. Truncation and ties-away controls disagree on 30720 and 15360 values respectively. An independent IEEE nearest-even bit-round calculation matches original. `cpu_result.json` preserves measured epochs/counts.

Initial task `task_20261108_013152_33218937137` completed the CPU stage then failed before compilation because this helper incorrectly passed a chip string directly to RunConfig. The helper now uses the repository's `golden.normalize_config`; the first log remains `task_submit_attempt1_config_error.log`. This was a helper error, not a compiler/runtime failure or bypass.

## Signature compile and generated code

The successful task imported the exact proposed and old modules and called only `module.hc_post_test.compile(config=cfg)` with **zero tensor arguments**. No samples, input preparation, worker/prepare or device calls. Both generated artifacts use **a2a3 / BackendType.Ascend910 / ExecutionMode.ONBOARD**. Metadata preserves dynamic T, BF16 sublayer, FP32 residual/coefficient/output contracts. Source/compiled/environment pins are retained; 140 compiled files occupy about 2MiB.

Candidate `compiled/candidate/ptoas/hc_post.pto:154-156` contains explicit f32→bf16 `RINT` then bf16→f32 `ROUND` conversions; repeated tail form appears at `:184-186`. Generated C++ `kernels/aiv/hc_post.cpp:650-663` retains `CAST_RINT` / TCVT into BF16 then expansion into float. `generated_cast_evidence.json` includes both pipeline copies. This proves lowering preserves the conversion, **not hardware rounding behavior**. Old-control C++ has no final CAST_RINT. No compiler fault or runtime patch is asserted.

## Concrete later native gate

`run_native_AFTER_ROOT_GO.sh` and `probe.py native --root-go` are ready; exact queue command is in `NEXT_NATIVE_COMMAND.txt`. Do not submit until explicit Root GO after the shared-owner native2 + post-audit seal. The helper requires exactly one allocated TASK_DEVICE, rechecks all source/compiled hashes, rejects SIM through ONBOARD normalization, and reuses both genuine compiled wrappers through the existing golden harness.

Five tiny fixture rows (`T=5,H=4,D=5120`) exercise the actual proposed operator, not a mirror. Expected values come from the pinned actual original helper. The gate delegates ordinary comparison to the existing HC-post `ratio_allclose(atol=1e-6,rtol=1e-5,max_error_ratio=5e-3)` and additionally requires **bitwise equality plus every FP32 output exactly BF16-representable**. There is no tolerance relaxation. The old genuine compiled producer runs against the same boundary gate and must produce a numerical failed receipt with positive mismatch/nonrepresentability counts. A runtime/compile error or missing receipt cannot count as a successful negative control. Two finite dispatches, one card, 64MiB ring heap; no full model, real weights or text generation. Native correctness/memory peaks are not claimed before execution.

## Handoff

All files are in this NEW owned directory. Shared writes=0; no source application, NPU/grant, old-unit edits, runtime/dependency changes, issues or external messages. Measured start/end are in work_record; helper/commands/source/compiled pins/report are sealed by artifact_manifest/PARK. Root imports once. PARKED at a compiled, reviewable native-preparation milestone; **native acceptance and text causality remain open**.
