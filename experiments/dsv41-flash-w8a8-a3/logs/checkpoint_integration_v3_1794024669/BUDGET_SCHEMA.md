# Concrete stage-budget contract (no REAL grant supplied)

The V3 CLI's root authorization must bind current pipeline/store/source/helper
and helper dependencies. For validate/decode/validate-chain it also binds
reference_policy_sha256 and derived_receipt_sha256. The referenced measurement
file must match large_io_budget.measurement_receipt_sha256. These files require
separate root assignment; this document is not a measurement or approval.

The measurement JSON schema is `split-stage-measured-budget/1`:

- measurement_kind: MEASURED_REAL_SPLIT_VALIDATION (CPU mock tests instead use
  CPU_MOCKED_GRANT_ONLY, never interchangeable).
- units: bytes; stage: exactly the assigned CLI stage; provider:
  REAL_FROZEN_SYNTHETIC (mock tests CPU_MOCKED_FAKE_ALLOCATOR).
- measured_at_epoch: actual integer timestamp, not future, at most3600s old.
- source_manifest_sha256: ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.
- META_geometry: expert_layers1,static_arguments57,static_bytes_per_rank6463879104,
  prefix_nonstatic_bytes7609876736,decode_nonstatic_bytes247111680,prefix_tokens128.
- required_bytes: host/disk/shm integer byte requirements including justified
  stage plan and margins. measured_available_bytes: same three fields from
  actual measurement. All recorded and fresh values must meet required values.

The code independently rereads MemAvailable, work filesystem statvfs available
bytes and /dev/shm statvfs available bytes BEFORE fixture factory invocation.
Query errors/unknown schema/source/stage/provider/units/time fail closed. The
actual public entry META contract is checked again before materialization.
There is no identity-capacity168 arena-bank reserve or NPU preparation here.

Let P=7,609,876,736, D=247,111,680 and G=111,031,942,400 global logical source
fixture bytes (16*6,463,879,104 static plus P). The implemented conservative
SOURCE FLOORS, not measured physicalRSS or a sufficient fit proof, are:

| Stage | Host floor | Additional free disk floor | SHM floor |
| --- | ---: | ---: | ---: |
| static-plan |2G=222,063,884,800|67,108,864|0|
| prefix |2G|2P+67,108,864=15,286,862,336|2P+13,421,772,800=28,641,526,272|
| validate |2G+2P=237,283,638,272|67,108,864|0|
| decode |2G|8D+67,108,864=2,044,002,304|28,641,526,272|
| validate-chain |2G+2P|67,108,864|0|

Fresh disk free already excludes existing checkpoints, so decode's additional
floor does not charge the retained prefix a second time. Retained logical
payload minimum after prefix+3positive+1control is17,196,646,912B before headers,
filesystem/atomic staging and explicit margins. Partial pending directories
also consume free disk. Do not equate logical storage with physicalRSS.

Store now explicitly deletes the copied serialization payload dictionary
before constructing the next one; weakrefs test this lifetime. Each clone has
self-contained storage, including views of large backings. CPU reference
clone/transformation/transient allocations and IO/runtime costs still require
an actual measured stage plan above these floors. Prior136GB PSS is a previous
run observation, not a prediction of current fit. Native phase HBM/owner/ring
proof and full-weight fit remain separate; this gate does not authorize them.
