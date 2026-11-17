# Independent typed F09 signature review — FINAL/PARK

Disposition: **PASS for the reviewed source migration and three zero-sample public signature compilations only.** Native routing/count arithmetic, captured task graphs, scheduler liveness, state transport, 19 numerical gates and general F09 remain unaccepted. The separate adapter lifecycle repair is outside this frozen207 review.

Reviewed Main FINAL207 `../f09_typed_completion_fix_1794079141`; artifact manifest SHA256 `08991711288e13970e27c8b39f5bf0ceb6a2da564c01049275473acaf10ac241`; copied45 UNAPPLIED source `405e7857b76e4813a3bdd0ce6dd2175f09617034cd3cc9ca68906d02fbf6bb31`; proposed full patch against live M02 `f726051f3b332e1db19daee444919b39801df1b76e56d712622f1e258c8cf0d0`; typed delta against failed93 `f4dfab6461dc219d69d3ce4efa4520998286e655c60e0c5f6e9eb9c2a8e58eaf`. Independently shell-checked all207 artifacts/all45 copiedsource before imports, all OK. Driver rechecks every frozen artifact before/after each compile; all intact. No source/frozen files were edited.

## Source findings disposition

Both shared inline helpers now return one unconditional stable tuple with complete explicit Tensor + Scalar[TASK_ID] annotation. The polymorphic optional return flag is removed. All9 actual model call sites and both probes unpack consistently: FFN→primitive1, fullentry decode2/prefill2, standalone decode/prefill2, standalone layer decode/prefill2. Standalone public chip entries still return only the tensor; fullentry signatures remain unchanged. No duplicated FFN/wrapper or compiler changes were added.

This matches the pinned public inline tensor/TaskId example in pypto tests/ut/jit/test_decorator.py:4254 and the explicit nonTensor tuple element annotation rule in jit/specializer.py:1384–1454. The prior93 failed task031644 remains FAILED and immutable. No upstream fault is alleged.

All four prior fixes remain coherent in this source: legal initialized-window/expertcount anchors; positive physical capacity2 combine with inner active predicate and explicit completion→export; separately recorded fixed-independent unconditioned19 versus original staged19 verdicts using unchanged gates/tolerances; exact nonpublication C2 state parity retention; exact forced-destination/source/expert matrix before a specific oldphysical192-versus-runtime96 verdict. No new source blocker found.

Current HC kernel and layer reference retain the accepted BF16 RINT→FP32 post boundary. The typed delta changes return/call plumbing only. W8A8 math, topology TP4/DP4/EP16, state ownership, thresholds and public model ABI remain unchanged. Original semantic reference ancestry is inherited from earlier review: archived rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d; utils reference file577c667fbf8f8b6f20de8a0ac54c6c0435605aaee1f6c2ccc463b446055bd286 and original MoE/model file101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91. No original-framework computation was run here.

## Independent execution

Actual sole queued job `task_20261108_033145_97507819480` COMPLETE **exit0**, zero-sample compile driver elapsed66.455s. Three stages all passed:

| Stage | Seconds | Independent file manifest SHA256 |
| --- | ---: | --- |
| count_active | 9.955 | 4c226662c16675febc4eea21c4de345a79e6562dc078549df46185212cabfaba |
| count_old_physical | 9.286 | 484bbb7d6c22475edbb725186e59b9dc1ff32118edfac9fa2dc3a33afc36a414 |
| decode_capacity2 | 42.271 | 5262e68afa0b53291ed285e31170b3c017674c3c25dfa9aec78eb7fab5777532 |

Artifacts are under this owned directory's build/<stage>. Both probe ABI metadata SHA256 `189381005f808c601c93286c4ef8022e54e160f99a2c144d527608df90dd9fb9`; fullentry metadata `8ba379ca7bd864d5d452bc82f171bb1f1d4495fea074b7607c8ba25552e78770`. Independent active ABI matches Main's earlier metadata but these are fresh independent compiled outputs. Source and compiled file manifests bind each result.

Explicit config normalized ONBOARD/a2a3, distributed logical IDs0..15, ring[0,0,4GiB,0], copied model expert-axis E1, _MOE_ROWS2. Fullentry contains all40 dense layers with shared E1 synthetic expert-axis ABI. Physical EP capacity2 is compiled; T/B/page/state signature axes remain dynamic (-1), consistent with future T2/B2/fourpages/twostateblocks dispatch geometry. No B2 runtime dispatch or fixture construction occurred; do not describe signature compile as actual physical shape variation.

Public generate.compile_entry(...,specs=None,...)→entry.compile used no TensorSpec construction/create_tensor, sample tensors, model fixture, checkpoint/header/reader, CPU golden, public prepare or NPU call. Python imports/metadata objects only. No tiny fake suite was repeated. Three outputs are C++/PTO/metadata signature bundles, with native images deferred to public preparation; no claim of native executable preparation/link success.

Pin chain recorded in environment.json: pypto38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e, runtime31f9e72ee6832cd198b77df18c1c305e7fd83676, pinned ISA f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7; unchanged installed PTOAS0.72 matches toolchain pin. No environment modification.

## Generated proof and limits

See GENERATED_PROOF.md for exact independently emitted source paths and line anchors. Actual countprobe HOST emits every rank unconditionally. CHIP code reads token_counts[my_rank], derives active view extents and passes their dynamic token bound into the real shared route count/push tasks. Old-physical CHIP instead emits constant2 active views. Positive block_num2 combine, explicit combine→return_consumed→export TaskIds remain outside the optional HC-post predicate, so source generation does not depend on a nonexistent local0 output producer. Metadata publication has a generated expert_counts writer and a ready-task input on that same tensor, then explicit ready→combine→release→export edges. This is emitted code and TensorMap contract evidence; no runtime deps.json was captured, and task execution/reachability on a live local0 rank remains unproved.

Actual fullentry emitted num_tokens from token_counts[rank], both layer loops and both runtime FFN count sites, capacity2 combine tasks and explicit release edges, and HC output slice→parent reshape→snapshot/carry inputs. Raw fullentry ABI has the existing16 Out/InOut parameters (12 mutable plus4 pureOut); the existing semantic19 comparator outputs split/group these tensors. There is no fullentry recv_meta output. Numerical gate coverage19 is preserved in source, not executed here.

Compiler warnings include existing unused read-anchor names and conservative scalar cache-line sharing warnings. Successful signature compilation is not evidence of native race freedom or numerical safety. Runtime visibility, generated dependency graph realization, allrankEP service (middle rank8 local0 serves96), idle/growing cache state, oldphysical specific rejection, within-call independent/original parity and required negative controls remain native/offline gates.

## Recovery

Job fully drained at epoch1794079977.098 before Root's terminal-audit hold; no further CPU tasks/children were launched. Root notified immediately with exact task/status. Remaining work here is read/report only. Native permissions unchanged: no prepare/device allocation/capture/weights/acceptance/source application granted. Goal9 physical B/T variation, EOS/slot reuse and general dynamic support remain open; decision10's temporary Engram disable does not waive them. Live current M02/fullreal15 path remains unchanged and separate.

Main's adapter preprepare .bin guard is a known independently scoped next-unit hold; frozen207 is not claimed capture-ready. Review its later lifecycle revision separately. Preserve paths once compiled. Root imports unique measured work_record.json; shared writes0; FINAL/PARK.
