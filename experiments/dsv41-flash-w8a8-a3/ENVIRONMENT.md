# Environment record

Environment revision: 3 (current; verified 2026-11-05 on the realigned
 toolchain). Revisions 1/2 below are kept as history. Keep machine-specific
values in the local experiment directory; do not commit credentials, device
logs or private paths.

Same selected environment as the sibling `dsv4-flash-0731-native` experiment:
all verification below was executed in this session on this host.

## Target and allocation

| Field | Value / evidence |
| --- | --- |
| Target project and revision | pypto-lib @ ab0fcd333d71701d063db4b9218ad2422f891ec4 |
| Backend integration and revision | model-harness `integrations/pypto`; PyPTO checkout @ be23606f791fa2251d7fccabdb91c75582268fad |
| Platform (`sim` or device) | `a2a3` (Ascend910B-class). Simulator smoke test executed; device execution deferred until topology agreement and device allocation |
| Host architecture / OS | Linux aarch64 (kernel 5.10.0-182), Ascend NPU host, npu-smi 26.0.rc1 |
| Python executable and environment | `.venv-local/bin/python` (3.10), activated via `activate-pypto.sh` from the pypto-lib root (user-designated environment for this task) |
| Physical devices allocated | Host exposes 16 logical Ascend910 devices, 65536 MB HBM each (~2.9–3.1 GB baseline use, `npu-smi info`). No device allocation recorded yet; device runs not authorized |
| Requested topology and rank mapping | A3 form (user-agreed basis): DP4/TP4 + expert parallel on 16 logical devices; rank table to freeze at agreement |
| Experiment directory | pypto-lib `experiments/dsv41-flash-w8a8-a3` (untracked local storage) |

## PyPTO dependency pins

| Component | Source of truth | Selected value / evidence |
| --- | --- | --- |
| PyPTO checkout and commit | selected checkout | `code/pypto` @ be23606f791fa2251d7fccabdb91c75582268fad |
| simpler runtime checkout and commit | PyPTO `runtime/` submodule | runtime/ @ 6b6b88f2353cca12cfafe809750b50b59931b0d1; installed simpler 0.1.0 `direct_url.json` = `file://.../code/pypto/runtime` (pin-aligned) |
| PTOAS release and architecture | PyPTO `toolchain/versions.env` | pin `PTOAS_VERSION=v0.70` (aarch64); `$PTOAS_ROOT/bin/ptoas --version` = `ptoas 0.70` (verified) |
| PTO ISA commit | PyPTO `runtime/pto_isa.pin` | first line 217bb98952a8f9004b95fe938da97d8d5f68f702; resolved `PTO_ISA_ROOT` HEAD = same (match) |
| Target project source path | target checkout | pypto-lib @ ab0fcd333d71701d063db4b9218ad2422f891ec4 |
| `PYTHONPATH` | target project instructions | pypto-lib root (set by `activate-pypto.sh`); `from golden import run` resolves |
| `PTOAS_ROOT` | selected PTOAS installation | `code/ptoas-bin/v0.70`; `bin/ptoas` reports 0.70 |
| CANN/toolkit revision, when using devices | selected `set_env.sh` and host inspection | `/usr/local/Ascend/ascend-toolkit` offers 8.1, 8.1.RC2, `latest` with `set_env.sh`; sourcing deferred until device work is agreed |

## Preparation ledger

| Step | Command or source | Status (`inspected` / `performed` / `deferred`) | Actual result / artifact |
| --- | --- | --- | --- |
| Target setup instructions read | pypto-lib `AGENTS.md`, `.claude/CLAUDE.md`, `.claude/rules/models.md`; harness `integrations/pypto/README.md` | performed | read before this record |
| Existing environment and compiler inspected | `activate-pypto.sh`, `npu-smi info`, `ls /usr/local/Ascend` | performed | pre-existing `.venv-local`; gcc/g++-15 via `pypto-toolchain-bin`; CANN present |
| Python environment created or selected | `.venv-local` | inspected | reused as the user-designated environment |
| PyPTO submodules initialized | `git -C $PYPTO_ROOT/runtime rev-parse HEAD` | performed | runtime/ @ 6b6b88f2353cca12cfafe809750b50b59931b0d1 |
| CANN environment sourced, when required | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | deferred | no device execution yet |
| PyPTO installed from selected checkout | `direct_url.json` | performed | pypto 0.1.0 from `file://.../code/pypto` |
| simpler runtime installed from the selected `runtime/` | `direct_url.json` | performed | simpler 0.1.0 from `file://.../code/pypto/runtime` |
| Pinned PTOAS installed and exported | `activate-pypto.sh` (`PTOAS_ROOT`) | performed | `ptoas-bin/v0.70`; `bin/ptoas --version` = 0.70 |
| Pinned PTO ISA resolved | `ensure_pto_isa_root()` in `activate-pypto.sh` | performed | `PTO_ISA_ROOT` HEAD = pin (match) |
| Imports and dependency revisions verified | `python -c "import pypto, torch; from golden import run"` | performed | pypto from `.venv-local` site-packages; torch 2.14.1+cpu; golden OK |
| Integration smoke test completed | `python examples/beginner/hello_world.py -p a2a3 --sim` | performed | `[RUN] PASS (10.05s)`, exit 0 (simulator) |
| Device workflow verified | `task-submit --device auto --run '... attention.py -p a2a3 -d $TASK_DEVICE --tokens 16'` | performed | NPU device 2 allocated; `[RUN] PASS (7.88s)`, device_wall 409479 clk, exit 0 (implementation agent) |

## Verification and handoff

Actual commands and results (executed 2026-11-02 in the `activate-pypto.sh`
shell):

- `source activate-pypto.sh`; `which python` = `.venv-local/bin/python`.
- `python -c "import pypto, torch; from golden import run"` — OK
  (torch 2.14.1+cpu).
- `$PTOAS_ROOT/bin/ptoas --version` — `ptoas 0.70` (pin match).
- `git -C $PTO_ISA_ROOT rev-parse HEAD` = 217bb98952a8f9004b95fe938da97d8d5f68f702
  = first line of `runtime/pto_isa.pin`.
- `git -C $PYPTO_ROOT/runtime rev-parse HEAD` = 6b6b88f2353cca12cfafe809750b50b59931b0d1;
  installed simpler provenance points at that `runtime/`.
- `python examples/beginner/hello_world.py -p a2a3 --sim` — `[RUN] PASS (10.05s)`,
  exit 0.
- `npu-smi info` — 16 logical Ascend910 devices, 65536 MB HBM each,
  ~2.9–3.1 GB baseline use (inspection; no allocation claimed).

Not performed (deferred): sourcing CANN `set_env.sh` explicitly (the queue job
did not require it), full-scale device execution (only a bounded component
kernel was run), and distributed 16-rank execution. The bounded device run
closed the queue/device-workflow check; it does not close the complete-model
or generation requirements.

## Open environment findings

| Finding | Impact | Owner / remedy | Status |
| --- | --- | --- | --- |
| Bare `ptoas` in PATH resolves to `/usr/local/bin/ptoas` (0.67), shadowing the pinned `$PTOAS_ROOT` (0.70) | Workflows invoking `ptoas` by name use the unpinned 0.67 | Invoke `$PTOAS_ROOT/bin/ptoas` explicitly or prepend `$PTOAS_ROOT/bin` to PATH in run shells | open |
| No device allocation recorded; evidence is simulator-only | Device gates (allocated-device smoke test, real-device generation) cannot close | Agree topology, record allocated device IDs, source CANN `set_env.sh` before first device execution | open |
| Host = 16 logical devices x 64 GB; the A3 form's container view is 16 logical devices x 64 GB (8 NPUs x 128 GB) — aggregate matches, per-NPU naming differs | None functional for the model task; rank planning should target the 16-logical-device container view per the A3 form | Freeze the rank table on 16 logical devices; confirm device visibility inside the execution container before device runs | open |

## Machine-switch re-verification (rev 2, 2026-11-03, new host)

The user switched machines (/mnt/data and home are shared storage). Execution
evidence does not transfer between hosts; every check below was re-executed on
the new host by the observer:

- Host: Linux aarch64, kernel 5.10.0-182.0.0.95.r2220_157.hce2.aarch64; 16 logical Ascend910 devices x 65536 MB
  HBM (~2.9-3.1 GB baseline use, `npu-smi info`); CANN ascend-toolkit 8.1 /
  8.1.RC2 / latest with `set_env.sh` present.
- Venv chain intact: `.venv-local/bin/python` -> Python 3.10.22 (shared venv
  resolves on this host).
- `source activate-pypto.sh` + `python -c "import pypto, torch; from golden
  import run"` -> OK (torch 2.14.1+cpu).
- `/bin/ptoas --version` -> `ptoas 0.70` = pin `v0.70` (match).
- `git -C  rev-parse HEAD` = 217bb98952a8... = pin first line
  (match).
- `python examples/beginner/hello_world.py -p a2a3 --sim` -> `[RUN] PASS
  (10.87s)`, exit 0.
- `python -m ruff check models/deepseek_v4_1_flash_a3_w8a8/` -> all checks
  passed.
- Bare `ptoas` in PATH resolves to `/usr/local/bin/ptoas` (0.67) on this host
  too — same shadowing hazard as the previous host; always invoke
  `/bin/ptoas`.
- Device queue: the `task-submit-npu` skill is available on this host; device
  allocation on this host not yet recorded (device gates remain open until an
  allocated run lands here). Historical device evidence (attention component
  on NPU device 2) belongs to the previous host.

## Device execution on this host (addendum, 2026-11-03)

- All device runs go through the host queue: `task-submit --device auto
  --device-num N --max-time S "cd <pypto-lib> && source activate-pypto.sh && ..."`;
  the task shell receives the allocated ids in `$TASK_DEVICE` (comma separated).
  Observed 16-card allocation for the EP16/TP4 entries; 1-card allocations for
  standalone operators. Queue wait and device wall are reported separately in
  `TIME_LOG.md` where the tooling reports them.
- Example allocations/evidence: system smoke `task_20261103_104530_*` (16
  cards, codegen failure), `task_20261103_120216_*` (16 cards, decode-layer SWA
  PASS 99.42 s), `task_20261103_120534_*` (coherent layer suite),
  `task_20261103_114308_*` (1 card, operator sweep 2).
- Device inventory unchanged: 16 logical Ascend910 devices x 65536 MB HBM
  (`npu-smi info`); per-device HBM usage is visible as `HBM-Usage(MB)` in
  `npu-smi info`, so peak device HBM can be sampled during a run.
- Device-only findings recorded in TASKS (milestone A): two onboard codegen
  limits (int32 dynamic-row tile store; 1-column FP32 TLOAD layout), one
  device-arithmetic calibration for the double-quantized expert output, and the
  EP16 window-transport anchor defect (host-driver input written by the rank
  entry, compiler `[InParamWritten]`; fixed by read-only anchors).


## Environment revision 3 — user-directed toolchain realignment (2026-11-05, owner-verified)

The toolchain was realigned (observer-executed, user-directed). This revision
supersedes the pins in rev 1/2; the model directory survived byte-identical
through the pypto-lib pull (untracked), canonical hash unchanged
`712a9053c3b0a598289f6d59b391facfaad7ab158ef888a5a3ae28262fe9537d`.

| Component | Rev 3 value (owner-verified) | Verification |
| --- | --- | --- |
| Target project and revision | pypto-lib @ `b6d17d9fa7a1e3420fbb7ac3c108fee194ec06fc` (ab0fcd3 → b6d17d9f, 20+ commits; `golden/` unchanged) | `git rev-parse HEAD`; `git status --porcelain golden/` empty |
| PyPTO checkout | `code/pypto` @ `38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e` (be23606f → 396e1b6 → 38a8aa6; includes IR scratch-retirement/lifetime fixes, dynamic-type remap, pipeline scratch fixes, the explicit-rsqrt precision fix 8f088a5 (#3258) and canonical scalar SIMT helpers 38a8aa6) | `git rev-parse HEAD`; installed `pypto` metadata url `file://.../code/pypto` |
| simpler runtime | `code/pypto/runtime` @ `31f9e72ee6832cd198b77df18c1c305e7fd83676` | `git -C runtime rev-parse HEAD` |
| PTOAS | v0.72, pinned cp310 wheel (sha256-matched); `$PTOAS_ROOT/bin/ptoas --version` = `ptoas 0.72` | executed |
| `PTOAS_ROOT` | `$VIRTUAL_ENV` = `pypto-lib/.venv-local` (wheel install); `which ptoas` = `.venv-local/bin/ptoas` | executed |
| PTO ISA | commit `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7` (217bb98 → f83d4b7), resolved into the managed `simpler_setup/_assets/build/pto-isa` path | `git -C $PTO_ISA_ROOT rev-parse HEAD` |
| Python environment | `.venv-local/bin/python` 3.10, `activate-pypto.sh`; `import pypto, torch` OK (torch 2.14.1+cpu) | executed |
| Platform / devices | `a2a3`, 16 logical Ascend910 devices; validation via `task-submit` queue (decision 8) | queue |
| Topology | A3 form DP4/TP4 + EP16 over 16 devices | goal.md decision 5 |

### Findings recorded with this revision

- **PTO-ISA resolver clone timeout**: the resolver's internal 60 s clone
  timeout kills large clones without a proxy. Source `~/proxy.sh` for any
  network operation, or clone locally and place the tree into the managed
  path (this is how f83d4b7 was resolved here).
- **Runtime regression cured**: on the new stack our prefill prefix-1
  (embedding + SWA layer 0, T=2) runs to completion on 16 cards with no AICPU
  507018/507901, no hdc disconnect and no `comm_d0` leak; the earlier
  single-program runtime fault was a pypto-version regression, not a model
  defect (observer-verified; TIME_LOG 1793849126 and the 38a8aa6 verdict).
- **Final four-piece matrix (user-known-good)**: pypto `38a8aa6` + simpler
  `31f9e72` + PTOAS `0.72` (cp310 wheel) + PTO ISA `f83d4b7`, with pypto-lib
  `b6d17d9f`. This is the recorded acceptance toolchain; device evidence is
  tied to it.
- **Reference-model note**: upstream's "preserve rsqrt precision across PyPTO
  API versions (#1668)" is a reference-model precision-drift fix; our
  `rmsnorm`/rsqrt usage and comparator tolerances are to be re-checked on the
  new stack (no silent loosening; independent calibration evidence required).
- **Reference templates on the new stack**: qwen3_14b `decode_fwd` PASS
  (16.47 s); lowlat `decode_csa` PASS (407 s, 8 cards). lowlat is the live
  V4-Flash-family idiom reference; dspark/MTP remain broken upstream (stale
  `hc_pre` TileView parse failure at `38a8aa6`, predating the original pin) and
  are not used.
- **Precision treatment**: our `rmsnorm.py`/`hc.py` already pass
  `high_precision=True` to `pl.rsqrt`, matching upstream 8f088a5 (#3258). The
  observer-reported prefix-1 logits gap (2.36e-4 on a -10.11 logit) is inside
  the model's BF16 arithmetic floor: a 1-BF16-ULP perturbation of the head
  input gives a mean logit spread of 5.8e-2 (max 0.29) at |logit|~10 on the
  host. The fwd acceptance comparator for logits is ratio_allclose(atol 2e-2,
  rtol 2e-2, <=0.5% outliers); the 2.36e-4 diff passes with ~2 orders of
  magnitude margin. The stricter rtol/atol ~1e-5 check that produced the flag
  is below the BF16 accumulation floor and is not an acceptance gate.
- Device evidence recorded under the old toolchain is historical; the green
  set is being revalidated on this revision (12 mode entries, both layer
  suites, EP16 drivers, structural gates in a fresh build root).


### Build-input pin discipline (2026-11-05, root-cause finding)

The PyPTO Python binding is a nanobind extension. The pypto checkout's
`build-constraints.txt` pins the build inputs (the file states what pypto is
*built against*, not merely compatible with):

| Build input | Pinned value |
| --- | --- |
| cmake | 4.4.2 |
| nanobind | 2.15.0 |
| ninja | 1.13.0 |
| scikit-build-core | 1.0.3 |

**Finding**: our original `.venv-local` (built 2026-11-02) used the
integration guide's UNVERSIONED install command, so build inputs were
unpinned and nanobind resolved to 3.1.0. On that binding, PyPTO's own
`ir.TileView()` zero-arg construction fails (`type_resolver.py:94`,
`_implicit_tile_view_defaults`) for tile-view-typed (NZ-annotation)
assignments — the cause of the dspark/MTP `hc_pre` parse failures on this
machine. `build-constraints.txt` documents that nanobind 3.0.0 broke upstream
CI on every branch on 2026-08-24 (#2492/#2493); a clean build with the pinned
inputs is required. The environment is being rebuilt (pypto + simpler, cache
cleared, nanobind 2.15.0) and will be re-verified before further device
validation.

**Process improvement**: the harness `integrations/pypto` install command
should apply `PIP_CONSTRAINT` from the checkout's `build-constraints.txt` (or
pin the four inputs explicitly); the unversioned command is how the violation
occurred.

## Recovery record (2026-11-06, observer): per-chip simpler_init failure after a 507018 fault
Symptom: 16-card submissions die pre-dispatch at chip init (simpler_init failed
with code -1000, consistently on chip 13; earlier chips 15/13), while npu-smi
reports all chips Health=OK at baseline HBM and 1-4 card runs pass.
Cause: leaked per-chip runtime state segments in /dev/shm from the faulted
run (sp-<runtime-generation>-chip-0..15 and 16 smp_* files, all timestamped
at the 507018 fault moment; clean exits unlink them, crashes leak them).
Remedy (user-level, verified): after confirming no live process holds them
(fuser), remove the leaked segments of OUR runtime generation (rm
/dev/shm/sp-<gen>-chip-* /dev/shm/smp_*). Segments owned by other users
(e.g. an Oct-28 generation) must not be touched. Post-cleanup verification:
16-card dspark decode_fwd --tp 4 --ep 16 PASS, all 16 locks released cleanly.
Do not confuse this with device faults: check /dev/shm for leaked state
before requesting an admin reset.
