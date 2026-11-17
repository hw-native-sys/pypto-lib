# Engram F05/F06 readiness

Read-only preparation completed before explicit implementation GO. No model source, acceptance artifact or NPU job changed. Original-reference CPU task task_20261106_172236_41627345879 exited0. New observer GO authorizes implementation without another approval round.

Pinned rc2 common.py:55–72 restores hidden FP32 with repeated block rotation transpose before original-basis channel-weight/key gating. Value projection already produces rotated V; residual stays rotated. Original model.py:1565–1592 calls Engram before attention. Checkpoint config/README/export agree: key/gate original, V/residual Quarot global, no runtime delta rotation. q*k per-channel folding stays valid after hidden basis restoration.

Proposed resource ABI: one global FP32 rotation[32,32], stored R orientation, restore h_rot @ R.T; distributed entries replicate[EP_SIZE,32,32], shared across Engram layers. Loader reads bounded optional/quarot.safetensors, verifies full repeated-block32 structure as original common.py:17, caches its block; never load full weights/tables to obtain it. Declared rotated checkpoint cannot silently get identity. Fixtures use explicit nonidentity orthogonal blocks; real paths use actual map. Existing INT8 group32 dequant and already-rotated Wv stay unchanged.

Worker Engram scope: engram.py only + artifacts. ABI engram_from_rows(lookup,wkv,weight,rotation,x,output). HC streams external FP32; restoration FP32. Independently ground original gate/HC cast boundaries before retaining/changing candidate BF16 narrowing; no golden-copy acceptance. Main owns loader,forward,layer,generator binding/reference files and required contract adaptation. Main moves Engram to layer input BEFORE HC pre and attention, preserving pre_mix semantics. No overlapping writers.

Basis controls: source-backed original gate plus analytic equation; signed/asymmetric keys,V,channel weights and nonidentity R; missing restore,wrong transpose,double V rotation. Preparation FP32 oracle rejects345/768,475/768,489/768 lanes at.02/.02. This is not candidate/cast/device validation. Owning defaults must exercise actual compute. Do not invent unsupported inactive-row claims; F09 remains open.

F06 controls: nonzero Engram delta and asymmetric attention weights; independently derive updated incoming stream and compare first actual HC/attention-input/query/cache boundary before FFN/MoE. Frozen post-attention placement must fail numerically. Layer1 SWA window and layer14 C2 compressor state/cache distinguish ordering. Validate prefill/decode actual consumers and full-forward compile bindings; no full-model parity from compile.

F07 can be disjoint: config.py,indexer.py,four full-source phase files,two C1 reindex files contain64-cap constants/defaults. A second worker can restore these while Main owns shared layer/forward/generator/loader files. Require actual>64/>2048 publication,cap-sized state,old64 negative and preserved source-pool consumption. Worker proposes shared-file edits to Main after park; no concurrent writers. Fixture resizing alone does not prove semantic restoration.

All CPU/NPU via task-submit,no simulator/commits/external filing/checkpoint/PyPTO writes. Final cohort freezes for sibling independent review. F09,F11–F13,offset incidence,full-model pool/state,TP/CI and generation remain OPEN/HELD. Readiness is preparation,not Engram acceptance.
