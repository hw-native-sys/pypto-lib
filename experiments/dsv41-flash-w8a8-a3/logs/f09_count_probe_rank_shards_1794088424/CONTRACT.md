# Rank-major diagnostic static contract

The logical HOST axes were ALREADY rank-leading in the previous count probes.
The six HOST pl.NZ layout tags caused public compilation to lower packed physical
leading axes: w1/w3/w2 became [(rank*24):rank*24+24,...], and shared weights became
[rank:rank+1,...]. Neither is a legal public StackedDeviceTensor leading integer.
This is an own diagnostic ABI mistake, not a compiler/runtime fault or missing API.

Change ONLY SIX HOST weight formals to default ND/pl.INT8 in BOTH count probes:
w1,w3,w2,shared_w1,shared_w3,shared_w2. Keep rank probe's local NZ annotations, all
CHIP code, actual moe_ffn/source405 and both HOST function bodies unchanged. This
matches the existing successful fullentry HOST-ND→CHIP-NZ pattern. Do not reshape,
repack, rewrite metadata, override __getitem__ or pass pointers around the public ABI.

Actual utils.pack_nz preserves the logical torch shape while reordering its bytes.
Therefore frozen producer f[name][:,0].contiguous() already provides precisely the
rank-major logical shape/dtype/canonical strides expected by new HOST ND. Local
CHIP NZ annotation owns physical interpretation. All SIX scale tensors were plain
FP32 rank-major already and stay unchanged; HC/gate/norm statics also audited.

EXACT static count18, not the earlier20 shorthand: HC3 +norm1 +gate2 +experts6 +
shared6. Selected bytes14,339,303,104 global /rank896,206,444 unchanged. The complete
static_contract_audit.json lists every fullE1 formal shape, selected layer0 shape,
dtype, guaranteed post-contiguous element strides and nbytes (both profiles). This
is a metadata/source derivation, NOT a huge fixture materialization/stride measurement.
A small actual pack/unpack nonsquare example additionally checks shape preservation.

New generated HOST selects w1 as [rank,0:24,0:5120,0:2304] and shared_w1 as
[rank,0:5120,0:2304], similarly all weights. The actual public StackedDeviceTensor
getitem method on metadata-only shard sentinels accepted every full slice for all
18 statics*16ranks*2programs=576 checks, and rejected the old leading-slice form.
No raw pointers/device handles/device preparation were used in this CPU check.

Two new count signatures/metadata/file manifests are in this unit. Fullentry artifact
remains EXACT old independent405 decode_capacity2, file manifest5262e68a…7532; no
fullentry recompile. compile_results.json explicitly mixes two NEW diagnostic rows
with one OLD unchanged fullentry provenance row; never relabel the latter as new.
Adapter count_profile/route signature-table paths now bind this exact mixed table;
all profile/role/source/compiled metadata validators and future Root scopes must use
new diagnostic manifest+probe SHA. Full reference171 policy/helperEE/oraclee45 remain.
