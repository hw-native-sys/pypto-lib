# Configurable A3 parallelism — initial bounded plan and ownership

Default remains TP4/EP16/DP4. Deployment uses WORLD=EP and contiguous TP groups,
DP=EP//TP. Proposed supported configuration domain: TP1/2/4/8, EP2/4/8/16,
EP>=TP and divisible; 384experts/64heads/8outputgroups/32indexheads and
129280vocab must partition legally with unchanged W8A8/NZ tile constraints.
This is a planned domain, not yet compile/native acceptance. Invalid combinations
reject before model imports/builders. Both --tp N/--tp=N and --ep N/--ep=N
are parsed at import time and accepted by executable CLI parsers.

## Initial matrix

| EP/WORLD | TP candidates | DP | Full real-E40 status |
| --- | --- | --- | --- |
|2|1,2|2,1|No fit claim; whole-expert EP residency exceeds device capacity|
|4|1,2,4|4,2,1|No fit claim; whole-expert EP residency exceeds device capacity|
|8|1,2,4,8|8,4,2,1|Routed weights alone 68.022435840GB/rank including FP32 scales; reject full run unless actual future proof|
|16|1,2,4,8|16,8,4,2|Default4 retained; TP8/EP16/DP2 first nondefault real candidate|

Header-free exact shape arithmetic will refine all memory numbers before any
real payload/NPU authorization. Routed whole-expert sharding is EP-based, not TP;
TP8EP16 still24localexperts. Dense/head/attention/indexer TP partitions change.
No old e13 projection or633997065216 global bytes is assumed for new topology.

## Exclusive writable scopes before dispatch

Worker parallel_config_collectives: config.py,tp_reduce.py,moe_exchange.py plus
worker_config/** only. Config public constants WORLD_SIZE=EP_SIZE,DP_SIZE=EP//TP;
existing import names retained, add shared CLI registration/parallel metadata
helpers as needed. Collectives/routed whole-expert ownership preserve semantics,
zero-active peer service/completion and default behavior; rename EP16-specific
function names consistently by notifying Main (Main migrates external callers).

Worker parallel_weight_loader: weight_map.py,weight_loader.py plus worker_loader/**
only. Preserve FP32 scales/offset-fold/NZ transpose/trueleaf consumer lifetime;
derive rank/TP/EP maps from synchronized config, no payload read or header sweep.
Main owns every other A3 model file, imports/callers/references, CLI propagation,
parameterized metadata/signatures/resource/Root admission and current overview.
No concurrent writers; request transfer before touching another scope.

## Milestones and meaningful checks

1. CPU config/invalid CLI matrix, independent rank/shard/NZ mapping and collective
golden small asymmetric inputs; exact default legacy equivalence. Metadata-only
E40 resource shapes/byte partitions and fresh signature compilation using ZERO
Tensor samples, no huge fixtures. Every check via task-submit --no-device.
2. Freeze coherent sources/evidence for Root-owned independent review, with
config support vs actual compile/native coverage separate. Root separately grants
small real-NPU nondefault EP communication/layer proof and TP8EP16 entry checks.
3. After separate resource/source/admission review, Root grants actual nondefault
real W8A8 noEngram generation using normal shared owner, one static upload and
strict DP/greedy/tokenizer/source/runtime/resource checks. No NPU or payload now.

Source baselineB30b/gen5de/newcd8b and all previous failed/accepted cases immutable.
A5 is syntax/reference only (its fullentry TP1 restriction is not A3 capability).
No quantization/reference tolerance/runtime/dependency/Engram/MTP changes.
Original parity/general dynamic serving/textquality gates stay explicit.
