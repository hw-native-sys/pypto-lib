# HC post BF16 boundary — unapplied draft

Main exclusively owns this NEW artifact directory. No worker. Live source45CC `cc15877f8f4b0104a842f915d2434d9a2c9cd6f6752d6364d5d7bada26e21d8c` and all old evidence remain unchanged. No product changes, compile, NPU, checkpoint tensor reads, large fixture or actual authorization.

## Proposed change

`proposed/hc.py` SHA `c6111634f78d53ad0cb4196a533a2d89097d4cfa0859688382c0072b60840885` changes only `hc_post` final accumulation store to FP32 -> BF16 `mode="rint"` -> FP32 and `golden_hc_post` to the same BF16 boundary. `proposed/layer_reference.py` SHA `c382bc7131c4bfd3f159d291c99de7e0d9f3368e5b25e1c3515b96f825dedda0` changes only `_hc_post_reference` final return. Patch `7520f4c6d7f144b0a2937751b9f43ad6c45c653c4b8fca49606c8d0dc4134c72`. The source45 proposed manifest is expressly UNAPPLIED, not a relabeled accepted source.

Public residual/output storage stays FP32. Required values are BF16-rounded, promoted exactly back to FP32. HC pre, HC head, signatures, accumulation/reduction sequence, callers, attention/FFN order, scale/quantization math and existing comparisons/tolerances are unchanged. AST restoration after removing only the cast pair and restoring the original store/doc is identical to the actual old producer. Every other function definition in both files is AST-identical.

## Original grounding

Original archive member prefix is `vllm-ascend-8fde6d4e335fe300e2b949aab0e31659cce0811d`; prior independently byte-verified archive SHA `48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`. `original_pins.json` pins the extracted original helper/model/proto/A3 kernel and independent review. Actual original `tests/deepseek_v41_utils.py:266` returns `y.to(x.dtype)`; function source SHA `a51b53d475db02841630c5d00ed18a3c7afb2de526962f0e0e59c7293f25abd5`. Proto outputs the BF16 sublayer dtype; selected A3 DSplit kernel uses BF16 `CAST_RINT`. Existing local HC head already uses `pl.cast(...pl.BF16, mode="rint")`.

Both actual old and proposed CPU functions execute unchanged via extracted AST, and expected output comes from the actual original helper. Original residual is BF16; proposed FP32 storage contains those same BF16-exact values. No new input rounding is introduced. This addresses the confirmed missing post-attention/post-FFN boundary (80 occurrences/40 layers), not a proposed extra arithmetic change.

## Main CPU self-check

Task `task_20261108_011604_328135122247`, COMPLETE exit0. Exact command:

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/hc_bf16_boundary_draft_1794071285/check_boundary.py'
```

Five 1x4x5120 cases: signed original reproducer, positive/negative even/odd BF16 ties, either side of midpoint, asymmetric nonidentity dyadic residual mix, and already-exact parity. Both proposed goldens match the actual original BF16 result promoted to FP32 bitwise in every case; inputs remain unchanged. Independent finite IEEE bit-rounding calculation also agrees. Old missing-round bad counts respectively 15360,20480,20480,5120,0 out of20480. Truncation rejects in four cases; ties-away rejects in both exact-tie cases. Signed tie values exercise negative finite inputs. The old maximum difference is0.00390625. Current live45 hashes verified before and after. `results.json` preserves per-case counts, hashes and inheritance lists.

These are Main self-checks, not independent acceptance. No import/decorator lowering, signature compilation or native execution was performed under this unit's restricted GO. The kernel cast API matches an existing local pattern but compilation and device rounding still require independent proof.

## Next independent native microproof (future Root GO)

Review the two-file diff and run the tiny CPU suite independently first. Then in a NEW source-bound reviewer artifact, compile only the ordinary `hc_post_test` consumer of proposed `hc_post` for a2a3 with `execution_mode=False`, preserving `[T,4,5120]` FP32 public output and BF16 sublayer input. Use the same signed/tie/nonidentity dyadic fixtures, original helper expected outputs promoted to FP32, and an additional bitwise equality check specific to the discrete BF16 boundary. The existing default tolerances remain unchanged; no tolerance relaxation is proposed. Capture generated a2a3 path, normalized ONBOARD mode and native runtime logs. Reject SIM.

Run the old missing-cast producer as a numerical control against that same original expected tensor (signed reproducer15360 mismatches), not merely a source spelling check. Proposed producer must have zero bitwise mismatches on all selected cases. Dyadic fixtures avoid reduction-order noise; no wider arbitrary-sum equivalence is presumed. A future selected ordinary model entry check is separate from the microproof. No real weights or full model execution is necessary for this first native gate. Reviewer command shape AFTER explicit authorization and creating that bounded helper:

```bash
task-submit --device auto --device-num 1 --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python <NEW_REVIEWER_HC_MICRO_HELPER.py> --platform a2a3 --execution-mode ONBOARD --source-manifest <NEW_BOUND_SOURCE45.json>'
```

The placeholders are intentionally future reviewer-owned paths; no such native helper or completed capture is claimed here.

## Application and remaining limits

Any application intentionally changes two math files. Construct a NEW actual45 manifest on the then-current source and independently rebind the accepted reference/math sourceplan, original equivalence receipt and generation admission/actual recipe. Existing 44-math equality checks against old `3dd67`/611 must reject this change until a new reviewed binding exists; never weaken them or relabel the old accepted source. The shared-owner generate draft and F09 draft remain separate and untouched.

Repetition causality remains UNPROVEN. No original/local real activation differential, corrected text run, full original-model parity or quantization parity is claimed. Weight reuse retains priority for the next live application after its own native proof. PARK at unapplied HC semantic draft + CPU self-evidence.
