# TP8 head padding focused review and CPU preflight

READY for separately authorized one-card original attention numerical validation; no NPU or numerical verdict in this unit.

Inspected v3 attention SHA562cda314635c65823e639d02a7e8db7100a65ef6d7006ff50bbd608d72e0cf7. The initial source snapshot and intermediate v2 are retained separately. Known prior failures were TP8 score16x16 versus mask8x16 (Main120530), followed by v2 sink fillpad outside in-core scope (Main121110). They are own implementation errors, not numerical/upstream results.

At attention.py:60–67 v3 loads/pads query and sink inside tokenSPMD. All added arithmetic, mask and state operations use HEAD_TILE16 inside that scope; reductions operate over keys independently per head. Query logical-valid8 rows are zero-padded before extending validity; sink logical-valid8 is likewise zero-padded. At lines103–105 final slice/store excludes padded heads and uses logical HEADS8 token stride. Original golden_attention_core and fixture builder AST match the initial snapshot exactly. Padded heads cannot affect live heads through a reduction. No blocker found in this bounded source review.

Other TP8 extents: o_proj.py token CubeM remains T_TILE16 with explicit token tail padding and logical stores; LOCAL_O_GROUPS1 changes its group loop/N-width, not CubeM rows. mla_proj query width4096 remains tile-aligned. Indexer retains32 replicated heads. LM head has existing explicit vocabulary valid-column tail handling. This is a finite source check, not full-model numerical acceptance.

ONE queued CPU task task_20261108_121806_13687553828 completed exit0: core16, packed640 and window128 original public entries compiled with ZERO tensor samples, normalized ONBOARD, a2a3 backend. Generated code/config/meta and per-artifact file hashes are saved. No builders/prepare/native/real weights executed. Signature dynamic T does not by itself prove runtimeT3; future invocation.json plus concrete CLI fixture shapes and runtime logs must establish that.

The finite native proposal uses native_cli.py to hash-check dependencies, require one actual TASK_DEVICE and invoke UNCHANGED attention.py CLI --tp8 --ep16 --tokens3 -p a2a3, --sim absent. Original golden and ratio comparator atol=.02,rtol=.02,max_error_ratio=.005 are unchanged. Output TensorSpec has no init_value, avoiding the known shared-output-initializer alias. Three cases contain 16/640/128 physical keys (multiples16); varying visible tails are 6/12/16,214/428/640,43/86/128. Random per-head sinks are included. This does NOT establish arbitrary physical nonmultiple16 keys or all-masked hardware behavior. No new compute entry/controller is proposed.

Planning resource floors HOST8GiB/DISK2GiB/SHM4GiB; largest case device tensors2,022,944B, not a measured runtime peak. One auto1/max600/client1200 request only after Root GO. No fixed physical card, production weights, distributed16 run or model text implied. Preserve first error and normal task lock release; do not retry hardware failure automatically.

This whole preparation tree is sealed. Future native outputs must be in a NEW exclusive tree via ATTENTION_RESULT_DIR; wrapper and source dependencies remain immutable. No source/model/runtime edits were made.
