# Configurable weight-loader/map worker — FINAL/PARK

Exclusive product changes: weight_loader.py and weight_map.py only. No other
product/shared documentation/ledger/A5/runtime/checkpoint files were written.

Actual loader sharding already used TP_SIZE/EP_SIZE/N_LOCAL_EXPERTS. Retained
whole-expert EP mapping independently of TP, packed source transpose/NZ order,
nonzero INT8 offset fold, FP32 scales, and trueleaf lifecycle. Added shared CLI
--tp/--ep registration and exact integer rank/tp-rank rejection before source
access. Removed fixed EP16/TP4 prose. Indexer32heads and shared FFN remain
replicated on every TP rank; TP8 does NOT halve those resources.

Corrected metadata-only mapping details: wo_a has local-output-group axis;
wo_b output width includes LOCAL_O_GROUPS; explicit indexer cases use correct
attn.indexer names and replicated full32head dimensions. Preserved existing
summary topology TP/EP/local_experts keys, adding DP/WORLD. Expert destination
pattern uses actual N_LOCAL_EXPERTS instead of24. No runtime/source-plan guard
or actual weight transform was changed. Frozen old weight maps remain untouched.
The per-key expert rank_local_bytes in build_map remains the existing amortized
source-size/EP accounting convention; real addressed whole-expert placement is
provided by WeightLoader and stream leaf source_names, not a divided tensor.

## CPU self-checks

1. task_20261108_112658_12111105507 --no-device/max300 COMPLETE0:
   check.py spawned13 valid topology imports, all physical rank identities,
   actual config-derived real model contracts, small actual-loader transform
   paths. Independent explicit NZ block iteration matches packed bytes; signed
   nonzero offset fold and FP32 scale slices exact; vocab/head/output-group
   slices correct across TP peers/DP replicas. Full384expert ownership partition
   per topology, expert streaming byte offsets/source names, pre-read invalid
   top-level contract denial and weakref leaf-release checks PASS. Expert payload
   shapes/group width were locally reduced ONLY in tests to avoid large arrays;
   actual production dimension assertions run first. No checkpoint reader used.
2. task_20261108_112819_122227522312 --no-device/max180 COMPLETE0:
   final preserved summary keys+DP/WORLD and replicated indexer map through actual
   build_map with fake metadata reader; fractional/overflow offset rejects;
   both loader/map public --tp=8 --ep16 --help parse with zero source reads.
   Final source hashes stable before/after. This checks the final summary-key
   compatibility refinement made while the earlier topology suite drained.

Commands and raw stdout are in task_v1.log/task_final.log; results.json and
final_results.json preserve outputs. These are self-evidence, not independent
acceptance. No signature compile, device prepare, NPU, real header/payload or
large fixture allocation was performed. No CPU failure occurred in these two
submitted tasks; no native/real-generation/topology-fit claim follows.

All owned product files now frozen for Main integration. Main handles all
callers, parameterized metadata/Root admission/source baseline, full entries,
independent review coordination and future native/real-weight authorization.
