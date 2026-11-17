# Native agent coordination

Effective 2026-11-06, explicitly requested by the user. This protocol supersedes
historical CLI-only, no-message-bus and no-subagent instructions. Existing repair
authorization, goal decisions1–9 and open acceptance rows persist.

The user's subsequent instruction delegates future acceptance runs to independent
subagents, superseding historical observer-personally-run requirements. Past
personally executed checks remain historical evidence.

## Roles

- Observer is the root. It owns unit scheduling, user decisions, dated Observer
  directives and acceptance decisions. It neither implements model fixes nor
  runs validation tests. It assigns independent reviewer/validator execution,
  reviews durable evidence and records the accepted scope.
- Main is the observer's native child. It owns implementation, integration,
  worker assignments and its own validation for one bounded unit. It parks at a
  coherent implemented/validated milestone, never claims observer acceptance.
- Workers are Main's native children. Before dispatch, Main records exact file
  ownership, interfaces, independent references, checks and stop conditions in
  TASKS. No concurrent writers may own the same file. Main retains shared
  integration files or transfers ownership after the previous writer parks.
  Workers report to Main and do not spawn further agents.
- Independent reviewer is a native child of the observer, outside Main's worker
  tree. It reports directly to the observer through messages and a durable
  report; it edits no model/golden/threshold files. Its review identifies a
  source manifest/snapshot and reference revision. Findings do not substitute
  for independent acceptance execution. When assigned as validator, reviewer
  personally reruns checks and reports actual commands, hashes, allocations,
  results and limitations to observer.

## Coordination and durable state

Use native messages, follow-up tasks and completion notifications for live
coordination; do not poll unit logs or start a new CLI session for each unit.
Messages supplement rather than replace these durable files:

- TASKS: latest dated Observer directives, active unit, named file ownership,
  commands/results, open gates and exact recovery state.
- TIME_LOG: append-only, real measured epochs, no invented intervals or closure
  of historical open rows. Preserve tags observer/main-agent; reviewer uses
  review-agent. Workers write measured work records with unique record IDs in
  their artifact directory; Main imports them once as worker:<task-name> with
  original epochs and source-record IDs. Parallel durations overlap and must
  not be summed as elapsed wall time.
- Each unit's artifact directory: source hashes, reference provenance, exact
  commands/task IDs/allocations/exits, per-output results, negative controls,
  report and resumption state. A message-only claim is insufficient evidence.
- COORDINATION_STATE.md: observer-owned live agent identities, unit/artifact
  paths, migration/ownership/acceptance state. Validate actual live identities
  after session loss; never infer runtime continuity from this file alone.

Serialize short TASKS/TIME_LOG append operations with flock on
`experiments/dsv41-flash-w8a8-a3/.coordination.lock`. Never overwrite a shared
ledger using stale read-modify-write text. Observer owns directive prose; Main
owns unit/ownership prose; reviewer writes its report and own ledger rows.
Workers do not edit shared ledgers directly.

## Bounded unit lifecycle

1. Observer records scope and launches/resumes Main. Main acknowledges latest
   directives, measures its start and records ownership before delegation.
2. Workers deliver scoped results to Main and park. Main integrates and runs
   checks; no file ownership transfer occurs while the previous writer is active.
3. At the coherent milestone Main stops unit-owned writes/jobs, records final
   manifest, evidence and recovery, messages observer and parks. A genuine
   external blocker requires a concrete checkpoint, not a pass or fake close.
4. Reviewer reports directly to observer. During independent acceptance no
   implementing agent changes the reviewed manifest. An observer-assigned sibling
   reviewer/validator reruns meaningful acceptance and negative controls on that
   frozen manifest, reporting directly to observer. Observer reviews results,
   records accepted scope/failures and assigns the next unit. Main self-tests or
   Main-owned worker checks cannot replace this independent acceptance role.
5. On session loss read TASKS/TIME_LOG/COORDINATION_STATE/latest unit report and
   goal decisions, verify source/process/job state, then create replacement
   native agents if necessary. No fresh CLI launch is required by this protocol.

The current native runtime exposes eight concurrency slots including observer
(2026-11-08 environment). Recheck actual available slots after recovery rather
than assuming a historical capacity. Prefer useful disjoint work, not
delegation solely to fill slots.
The user explicitly authorizes parallel nonconflicting bug fixes. Main can own
one repair and delegate other repairs with exclusive file scopes. It integrates
their results into a coherent cohort; every writer parks before independent
acceptance freezes that cohort. Shared-file repairs require serialized ownership.
Intermediate scoped tests during parallel edits must not claim a stable whole-tree
manifest; independent acceptance uses the final frozen integrated manifest.

## Migration and execution discipline

Let the existing CLI Main finish its coherent unit. Until STOP and process exit
are verified, native Main may prepare recovery/ownership read-only, but cannot
edit model files, launch implementing workers or duplicate acceptance jobs.
Observer-assigned independent validation checks future CLI milestones before transferring model-write
ownership to native Main. Do not kill a healthy unit merely for migration.

No commits, external filing or writes to checkpoint/PyPTO checkout. All NPU
execution uses task-submit automatic allocation; CPU/compile uses --no-device.
Source activate-pypto.sh before Python and proxy.sh before network work. Use
pinned PTOAS explicitly. No simulator acceptance, relaxed gates, fabricated
closure or candidate-oriented golden changes merely to obtain a pass.

After the proven omitted-mode defect, every device helper explicitly selects
ONBOARD (execution_mode=False or its enum), records normalized mode, generated
a2a3 platform and native runtime evidence, and rejects accidental simulation.
Queue allocation and requested backend alone never establish onboard execution.
