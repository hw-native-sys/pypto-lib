# Launch inputs (local; do not commit)

HARNESS_ROOT=/mnt/data/pyptouser/yangyaodong/code/model-harness
PROJECT_ROOT=/mnt/data/pyptouser/yangyaodong/code/pypto-lib
INTEGRATION=integrations/pypto/README.md
EXPERIMENT_DIR=/mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3
MODEL_ROOT=/mnt/data/models/DeepSeek-V4.1-Flash-w8a8 (read-only; completeness verified)
MODEL_ID=DeepSeek-V4.1-Flash-w8a8 (Aurora export 2026-09-10; early preview per checkpoint README)
ENVIRONMENT=.venv-local via activate-pypto.sh (pypto be23606f, simpler 6b6b88f2 from runtime/, PTOAS v0.70, PTO-ISA 217bb989)
REQUESTED_TOPOLOGY=A3 form: DP4/TP4 + expert parallel on 16 logical devices (host exposes 16 x 65536 MB)
INITIAL_SCOPE=W8A8 as shipped + INT8/BF16 Engram (contradiction open); text first (image follow-up proposed); DSpark/MTP excluded per A3 colocated form (confirmation open)
SPEC_BASIS=vllm-ascend tutorial rev vLLM Ascend v0.30.0rc2, section 5.1 (A3 single-node colocated)

The agent prompt is `HARNESS_ROOT/prompts/PROMPT.md`, resolved against
HARNESS_ROOT. Worker delegation via `HARNESS_ROOT/prompts/WORKER_PROMPT.md`
requires explicit authorization. Template links to the model rules were
rewritten to `HARNESS_ROOT/rules/models.md` per the harness README.
