# Configurable metadata worker — FINAL/PARK

NEW artifact-only scope worker_metadata/**. Previous worker_loader bundle and
all product/shared/old source files stayed read-only. No workers spawned.

metadata_plan.py accepts a mandatory externally supplied current45 source
manifest+SHA, exact cd8b reference plan, new nonexistent output, and --tp/--ep
both syntaxes. It verifies exact45 pathset+current hashes BEFORE imports and
again after tracing. It constructs actual build_entry_contracts(3,pages2,batch1)
with E40 expert axis and obtains current parallel_operation_budget(...,16).
No TensorSpec.create_tensor, checkpoint index/header/tokenizer/payload reader,
fixture, prepare, signature compiler or device is called.

MetaReader uses immutable old source_descriptors JSON only, returning meta
weights/scales. Actual WeightLoader.iter_argument_leaves drives rank/argument/
layer/expert order, source read counters, dtype/shape and contiguous target byte
coverage. Meta-only offset folding bypasses its Boolean reduction while retaining
all source offset reads; the small actual-value tests are separately preserved
in worker_loader. WeightLeaf CPU-device constructor checking is replaced only
inside this metadata process. No numerical/value or native proof is claimed.

One authorized queued CPU data-generation task:
`task_20261108_114844_12733984717`, --no-device/max900/client1200, COMPLETE0.
trace_pair.py serially ran TP8/EP16 then defaultTP4/EP16 against provisional
Main source_candidate_v1.json SHA
651580ea4f88a065a91182faa455b361278dc10dd6b739ed499c9f5545dac23d.
Both traces confirmed before/after exact source hashes unchanged.

TP8/EP16 metadata_tp8_ep16.json SHA
0b42980f2f7c4ea8bfc237683e930ae3e92fc1f6220563da02d275cb7f36800f:
38,328,513,216B static/rank,613,256,211,456B global,28.578s subprocess wall.
DefaultTP4/EP16 metadata_tp4_ep16.json SHA
4c4f3847ee569224ba7709ad99b69f4b94a11605425a823c3944825e1b808fa4:
39,624,816,576B static/rank,633,997,065,216B global,27.946s subprocess wall.
Each phase, both profiles:57 static names/912 rank targets,110368 leaves,
164192 source readcalls and745846365184 cumulative source bytes. Metadata
records unique-source counts/bytes, exact read count maps, complete rank target
ranges, generated zero bytes and parameterized conservative resource budget.
Source read amounts remain full-source physical-reader schedule, not selected
slice sizes. Decode shared consumer accounting is zero new static reads/upload;
the separately traced decode leaf schedule is only the borrow-layout equivalence
basis, not a second actual transfer. No new e13 equivalence claim.

All existing checkpoint/index/tokenizer/prompt descriptor fields are copied
verbatim from accepted cd8b, without rereading any checkpoint file. New schema
configurable-full-real-generation-metadata/1 binds exact current45 and topology,
with cd8b explicit parent_reference_plan_sha256. No positive runtime receipt is
created; upload/forward peaks remain null. Independent acceptance and actual
nondefault communication/full generation remain separate Root stages.

No failures in the submitted metadata data task. Task logs/progress/results are
retained. No Main/Root ledger writes; new unique worker-agent measured record
ready for Main import once. All owned files/writers/jobs PARK.
