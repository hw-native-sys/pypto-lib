# Exact SourceB auditor migration map

`input_sourceB_audit.py` and `input_sourceB_shared_delta.py` are byte copies of
the frozen1794099369 preparation. They are historical inputs, not executable
future audit entry points. Their READY pins and old task/case are not copied.
The4638/5 SourceB result remains immutable NOT_ACCEPTED218; no old case rerun.

| Historical source anchor | Required replacement / preserved gate |
| --- | --- |
| audit61, pins/sourceB handoff schema; audit86-97 | Fresh producer schema, operation, ALL45+plan+metadata-binding+math-lineage+signature descriptors; no old source30b/cd8b/A8153/historical704 as current. Historical HC/F09 numerical lineage stays separately labeled. |
| audit109 ordered16;115 config;125 normal CLI | WORLD ordered allocation, exact explicit --tp/--ep propagated before imports to actual CLI/reexec/compiler; new exact RootGO/task/case, ONBOARD/a2a3, ring policy and cap bound to new producer. |
| audit144 identity evidence shape,145 resource floors | New topology/resource/profile receipt fields, not a renamed old identity receipt. Preserve source/tokenizer/library PREQUEUE+allocated+POST/externalPOST. HOST64GiB/DISK2GiB/SHM16GiB only if new reviewed profile retains them. |
| audit181-184 canonical trace; delta6-10 constants | Fresh verified plan/source read counters; primary prediction global613256211456, reads164192/source745846365184. Never use old633997065216 or predicted counters as actual trace evidence. |
| delta30-95 bases/borrowed/close | Derive57*WORLD,7*WORLD C1,50*WORLD aliases from plan, then require actual exact coverage, zero-offset/full-byte equal-address same-owner views. Primary912/112/800. Decode static reads/bytes/H2D/allocations/copies remain zero. |
| audit211 memory events; delta96 finals | WORLD rank coverage for every actual dispatch and one actual closing phase. Shared owner has one WORLD worker set, two registered programs; no two-owner/prefill-free assumption. |
| audit218-225 greedy selection | DP=EP//TP votes, TP candidates/vote, vocabulary129280//TP span. Primary2DP/8TP/16160. Same finite, agreement, history, winner/tie behavior and score checks; no raw-logit/parity claim. |
| audit230-255 generation/state | PromptN3, cap16/EOS branch, actual prior token, positions/history, returned six persistent caches unchanged. Preserve all eight diagnostic prefixes when emitted; FLOAT32/SHA1[:16] qualification mandatory. No fixed15decode/old output IDs. |
| audit262-268 actual artifacts | New formal ABI and fresh topology signatures, not blind100/96 reuse. WORLD leading axis, w1 prefix[WORLD,40,384//EP], all fresh-plan static shapes/dtypes; TP8 shapes change even where w1 prefix stays[16,40,24]. |
| delta110-151 HC source profile | Current HC source-location/generated-symbol witnesses and fresh topology profiles, never old4/4PTO2/2CPP by default. Separate actual runtime compile artifacts from CPU signature dirs. Retain InOut6 caches and dynamic token binding proof. |
| audit280 workers;298-299 raw release/queue | WORLD native worker-ready identities, actual device/rank mapping, raw ordered WORLD release and zero occupied/idle queue status. No fixed total child count or invented complete process history. |
| audit151-166,285-295 strict telemetry | Unchanged FAIL on any incomplete inventory/memory/telemetry sample; whole PSS unknown stays null, partial sum is not peak. Native/CLI/outer status distinct; bounded facts never reclassify whole failure. |
| audit307 result baseline labels/327 scope | New source/topology identity; old sourceB math labels historical only. Actual raw text/IDs+repetition observation; quality UNASSESSED and original parity false. |

The pure-source `topology_delta.py` supplies the changed domain/rank/vote/plan
expectations and a fail-closed readiness gate. It does not replace the full
historical telemetry, lifecycle or source auditor. Integration into a NEW
actual-audit execution unit requires the final producer schema and a source
review before any separately authorized postterminal CPU run.
