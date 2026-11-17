# Independent emitted source evidence

Paths below are relative to this review directory. These are actual public compile outputs, not Main's fake TaskIds or inherited generated evidence. Line numbers identify frozen generated C++/Python. Runtime task IDs and runtime graph capture do not exist in this unit.

| Contract | Actual emitted evidence |
| --- | --- |
| Every EP rank dispatched | build/count_active/orchestration/host_orch.py:26–27 uses world_size and unconditional rank loop; lines52–84 pass explicit windows, per-rank tensor slices, token_counts and separate diagnostic output |
| Runtime count read | build/count_active/next_levels/probe_rank/orchestration/probe_rank.cpp:145–150 reads ext_token_counts[my_rank], casts active and constructs x_active view |
| Active flows into actual route count | Same CPP:327 derives tokens_inline1238 from active normed view; Task7 lines346–363 receives indices_active and dynamic tokens_inline1238, writes recv_meta and expert_counts |
| Count→push→data | Same CPP:346–398: Task7 count_tid→Task8 push deps→Task9 signal deps; compact Task11 lines410–428 explicitly depends on data_signal/count/clear |
| Metadata completion writer/read | Task7 add_output(expert_counts_inline1242) at352; Task21 moe_return_signal add_input(sameexpert_counts) at752, plus explicit return_tid fence. TensorMap RAW contract supplies metadata producer ordering; runtime realized graph unmeasured |
| Always-existing combine | Same CPP:943–956 submits Task30 block_num2 without active outer branch; explicit ready_tid input |
| Local0 release/export completion | Same CPP:958–969 release Task31 depends on combine_tid, yields return_consumed_tid/ep_done; optional hc_post predicate appears only after this; export Task33 at997–1007 explicitly depends on return_consumed_tid, outside hc_post predicate |
| Inner active guard | build/count_active/next_levels/probe_rank/ptoas/moe_combine.cpp:115 contains if(block index < dynamic token scalar); wrapper kernels/aiv/moe_combine.cpp:331–343 unpacks token scalar and active output shape before forwarding |
| Oldphysical distinguishes scalar path | build/count_old_physical/next_levels/probe_rank/orchestration/probe_rank.cpp:145+ emits view min(2,physicalextent), without active=get_tensor_data(token_counts[myrank]); same collective/completion structure |
| Real fullentry token read | build/decode_capacity2/next_levels/decode_fwd_rank/orchestration/decode_fwd_rank.cpp:233–238 computes TP4 group and reads num_tokens from token_counts[rank] |
| Both fullentry FFN runtime routes | Same CPP:4625/9742 num_tokens→active;4805/9922 active normed shapes→tokens; count tasks136/289 at4824/9941 receive these dynamic scalars |
| Both fullentry positive combine/release | Same CPP:5421/10538 capacity2 combine (Tasks159/312),5436/10553 explicit consumed after combine (Tasks160/313) |
| Slice publication into snapshots/carry | Same CPP:10570–10640 HC post receives active output slice, parent x_next reshaped to x_flat, snapshot315 reads x_flat, snapshot316 reads pre_next, carry317/318 write x_hc/pre_mix InOut. Analogous first loop at5479/5495 |
| Existing fullentry ABI | build/decode_capacity2/distributed_meta.json: six persistent cache/state InOut entries, four selection InOut, x_hc/pre_mix InOut, layer snapshots/final_hidden/logits Out; raw16 output parameters, original semantic19 gates in source; no diagnostic count parameter |

The source path count→ready (TensorMap producer/read)→combine→consumed→export (explicit TaskIds) survives local-active0 generation. Other protocol edges combine explicit deps and tensor argument direction/overlap inference. This establishes emitted dependency mechanisms, not a measured runtime graph path or scheduler liveness. A future native dependency-only capture must still verify tasks and edges on active/idle/zero-receive ranks, separate from level4 timing.
