# Future serving ABI

Spec revision: draft. This file defines interfaces only; serving is not executed.

The runnable model prefill/decode entries in `MODULES.md` are current required
deliverables. This file specifies how a future serving adapter will call them;
the serving exclusion does not defer implementation of those model entries.

- Inspected serving repository revision, or unavailable assumptions: the
  user-designated serving reference is vllm-ascend (tutorial rev vLLM Ascend
  v0.30.0rc2, DeepSeek-V4.1-Flash page). No local vllm-ascend checkout
  exists on this host; fields below record the documented A3 form as the
  reference contract. The local `pypto-serving` checkout @ fbfa125 is a
  separate stack whose docs mark this model "not supported"; treat it as a
  second reference only if the user re-targets serving to it.
- Model matching, capabilities, and limits:
- Prefill entry point and ordered arguments:
- Decode entry point and ordered arguments:
- Weight preparation entry point:
- Dynamic axes and compile-time versus runtime values:
- Token, position, length, mask, block table, and slot mapping semantics:
- KV page size, layout, dtype, ownership, allocation, and update semantics:
- Other persistent state: shape, dtype, ownership, initialization, and updates:
- Logits / hidden / token output semantics:
- Embedding and sampling location:
- Tensor lifetime, aliasing, completion, and buffer reuse requirements:
- Proposed compile/runtime argument builders and startup compatibility checks:
- Unsupported capabilities and integration questions:

## Documented A3 form (inspected reference, vLLM Ascend v0.30.0rc2, section 5.1)

Recorded so the future adapter contract matches the user-designated serving
form; serving execution itself remains outside this goal.

- Runner: model runner V1 (`VLLM_USE_V2_MODEL_RUNNER=0`; V2 unsupported for
  this model on Ascend); async scheduling.
- Parallelism: `--data-parallel-size 4 --tensor-parallel-size 4
  --enable-expert-parallel` on 16 logical devices; shared expert in DP
  (`enable_shared_expert_dp`); fused MC2, DSA-CP, FlashComm1 enabled.
- Limits: `--max-model-len 150000`, `--max-num-batched-tokens 8192`,
  `--max-num-seqs 32` (per DP engine), `--gpu-memory-utilization 0.9`,
  `--block-size 128`.
- Quantization/tokenizer: `--quantization ascend`, `--tokenizer-mode
  deepseek_v41`, `--reasoning-parser`/`--tool-call-parser deepseek_v41`,
  `--enable-auto-tool-choice`, `--safetensors-load-strategy lazy`.
- Graph capture: `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
  — decode captured, prefill eager on the serving side. The harness's
  complete-model compiled prefill/decode entries remain required model-side
  deliverables regardless of serving-side capture policy.
- Engram: `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`;
  container needs `--ipc=host` for the 4 local DP ranks to share host
  Engram memory (24 Engram heads do not split evenly over 16 logical
  devices).
- Environment: `VLLM_ENGINE_READY_TIMEOUT_S=36000`,
  `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `HCCL_BUFFSIZE=1024`,
  `HCCL_OP_EXPANSION_MODE=AIV`.
- Verification semantics (tutorial 5.6/6): `/health` -> HTTP 200;
  `/v1/models` contains `dsv41`; greedy text probe ("Who are you?",
  temperature 0, 256 tokens) returns nonempty content; image probe with
  `max_completion_tokens` 256. Note the tutorial's own anti-self-report
  stance (section 12.1): startup success or a true config field does not
  prove an execution path — output evidence is required.
- Limitations (tutorial section 11): production performance qualification
  and task-level accuracy evaluation are not complete for this form; PD
  decode uses DSpark + FULL_DECODE_ONLY, colocated 5.1 does not use DSpark.

## Scope revision 2 status (2026-11-03): these semantics are now implemented

The paged KV/block-table semantics defined above are no longer a future-only
interface: user decision 9 moves the model entries to the A5-shaped dynamic
batch/S interface with paged raw-window/compressed/index caches, per-request
`index_block_table [B_DYN, TABLE_DYN]`, `request_ids`, `compressed_lens`/slots
and `window_slots`/indices, plus runtime `num_tokens` and DYN axes inside one
compiled artifact. The migration is sequenced before complete-entry device
validation and generation. Serving execution itself remains outside this goal;
the ABI fields above become the implementation contract as the migration lands.
