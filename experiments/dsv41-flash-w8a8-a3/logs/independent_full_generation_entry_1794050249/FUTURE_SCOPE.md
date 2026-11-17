# Concrete next scope — proposal, not execution permission

Current review permits only a recommendation to Root. Product remains762dc. Apply only the reviewed v2 generate patch after Root assignment, preserve the other44 source digests, and independently verify the applied source and necessary signature-only E40 N3/decode1 artifacts. Never run the artifact copy as the producer or use a hidden callback/guard replacement.

The v2 proposed generate digest is `d057012464331b10b0bdc7f406925ee0c552dd298a1db171a93b0b1595769598`; whole patch `522dae74110e5e1725e9b36a399c6919138b1937d900a593ffc26ab3ca0f1dca`. `proposed_applied45_inventory_NOT_APPLIED.json` is only a planned inventory. Its contents must match actual product files after application before it becomes an actual source binding. Root's NEW task-local grant then binds that actual manifest SHA, unchanged metadata plan3dd67, map SHA, actual ordered16 devices, current installed library24434, operation and source/external-check code.

Required external tokenizer check, at all three stages PREQUEUE / ALLOCATED_PRE_EXEC / POST_EXEC:

1. Resolve both CLI model-root and checkpoint; require both equal `/mnt/data/models/DeepSeek-V4.1-Flash-w8a8` and the plan checkpoint.
2. Hash both exact tokenizer files and compare with plan.tokenizer_files (explicit hashes in REPORT). Retain root, actual paths, SHA, byte sizes, stat identity and timestamp in new operation evidence. Query failure or disagreement stops; do not silently refresh expected hashes.
3. Use a fresh normal CLI process so load_tokenizer cannot reuse an unrelated cached root. Require actual plain prompt IDs `[19923,14,2058]`, allactive N3, row2 and start0. Retain actual output prompt metadata and strict4DP votes/scores/history. Post-run hashes are required even if the CLI fails.

Normal command skeleton after separate Root payload/native authorization (max-new-tokens is exactly the separately granted2 or4):

```bash
task-submit --device auto --device-num 16 --max-time 21600 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && DEEPSEEK_PER_LAYER_EXPERTS=1 python models/deepseek_v4_1_flash_a3_w8a8/generate.py --device-run --device "$TASK_DEVICE" --backend-type a2a3 --model-root /mnt/data/models/DeepSeek-V4.1-Flash-w8a8 --checkpoint /mnt/data/models/DeepSeek-V4.1-Flash-w8a8 --template plain --prompt "Hello, world" --start-position 0 --rope-profile auto --handoff-offset-c2a 0 --handoff-offset-c1a 0 --engram-mode disabled --max-new-tokens 4 --repeat 1 --max-time 21600 --first-real-generation-authorization NEW_TASK_LOCAL_ROOT_RECEIPT --result-json NEW_OPERATION/generation.json --log-file NEW_OPERATION/generation.log'
```

The outer operation runner performs the admitted external checks and writes the grant only after actual allocation. It must invoke this ordinary product CLI, preserve its raw exit and first error, and perform postchecks without changing model execution. The illustrative command itself is not a grant. No --execution-mode / padding / shared expert / decode-state / alternate prompt or case input is allowed. Installed public normalized mode must be ONBOARD, platform a2a3, ring4GiB on prepare and each call. Use actual prior greedy outputs and ordinary returned state at positions3/4/5, not this review's tiny fake token history.

Source/copy/read plan is exact, not an optional performance target: per phase164192 reads /745846365184 logical source bytes /633997065216 completed H2D bytes /912 complete targets. Early prefill EOS may avoid decode; a phase that proceeds to forward must complete its own exact plan. Keep the v2 complete read map/endpoints and both raw final memory receipts. Root's explicit HBM observation-gap cadence and reviewer_fit's floors/cleanup controls apply; those numbers are PLANNING bounds, not measured full-forward peaks. No extra payload/NPU or large allocations were authorized by the current unit.

Stop on source/header/map/index changes, unrepresentable offset, incomplete source/target coverage, strictDP disagreement, budget/query/deadline error, native failure or close failure. Preserve first error, all partial progress and cleanup errors. No blind resubmission, heap change, E1 fallback, uninitialized forward or actual-text claim from the synthetic acceptance.
