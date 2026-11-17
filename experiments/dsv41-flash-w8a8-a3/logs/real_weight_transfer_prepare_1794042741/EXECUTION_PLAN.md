# Future initialized transport scope — NOT a payload/NPU authorization

Current unit used ONLY real header/index metadata and owned tiny CPU fixtures.
No real tensor payload, device prepare, large allocation, upload or forward ran.
Root must separately assign independent review and any actual execution.

## Fixed selected source/target table

All full57 real40 empty handles remain co-resident (912handles,
39,624,816,576B/rank) while these selected intervals are initialized/read back.
No other interval may be read and NO model dispatch exists.

| Source | Source shape/dtype/bytes | Device target and selected interval |
| --- | --- | --- |
| embed.weight |129280x5120 BF16;1,323,827,200B|4TP shards32320x5120,330,956,800B each at embed_weight offset0; each copied/readback on4DP peers, all16 ranks|
| layers.1.ffn.experts.1.w1.weight |2304x5120 I8;11,796,480B|rank0,w1[40,24,5120,2304], layer1/local1 prefix, BYTEoffset294,912,000; only11,796,480B|
| same base weight_offset |2304x1 FP32;9,216B|CPU q-offset fold input only, not separate device argument|
| same base weight_scale |2304x1 FP32;9,216B|rank0,w1_scale[40,24,2304] FP32, BYTEoffset230,400; only9,216B|

Embedding reads full source ONCE perTP, then reuses that source view for4DP
consumers before release; never16 separate source reads or globalrankstack.
TP source rows are[tp*32320,(tp+1)*32320); targets tp+4*dp. The expert key is
preselected layer1/EP0/local1=global1; direct expert_tensor/expert_scale avoids
reading25 preceding leaves. No search through real payload to pick a case.

Data-read allowlist/schedule/header/range/dtype/caps are checked BEFORE file seek
and read. Header prefix/JSON SHA, index SHA, file size/mtime, exact key/range and
absolute8+header+data_offset conversion are pinned. All source opens are rb;
no checkpoint write. Selected source fingerprints are computed only after a
future authorized read and saved as hashes, not tensor payload files.

| Counter/cap | Bytes |
| --- | ---: |
| Unique selected source key/ranges |1,335,642,112|
| Cumulative returned source tensor bytes (embed read4x) |5,307,123,712|
| Completed initialized-device transfer bytes |5,307,114,496|
| Exact D2H readback bytes |5,307,114,496|
| Per public copy chunk |67,108,864 (64MiB)|

These are logical API bytes, not physical disk/cache traffic or throughput.
Offsets are BYTES, including FP32 scale. Existing PreparedDispatch._upload_leaf
is the actual H2D consumer. D2H start is independently derived from ACTUAL
compiled57 ABI C strides and dtype-byte width; it is not reused from writer's
(layer*24+local)*leaf_bytes variable. Both must agree BEFORE payload work.
Scale prefix(1,1,0)=57,600 elements*4=230,400B; treating57,600 as bytes rejects.

Independent source expectations hash the original embedding row slice and
verify public CPU source-view address. Quant expectation independently computes
integer q-offset with representability checks and traverses source blocks in NZ
byte order (no loader pack_nz call); scaleFP32 bytes are preserved. Real offsets
may be allzero: log actual nonzero count, do not claim real nonzero-fold proof.
Tiny CPU nonzero offsets separately reject discarded-fold and wrong-NZ paths.

## Public interfaces and proof boundary

Installed DistributedWorker.copy_to/copy_from use nbytes/dst_offset/src_offset
in BYTES and bound the full allocation. Public DeviceTensor.data_ptr is the
registered base handle; helper does not access private runtime pointers or add
a kernel. A tiny spy executes the actual installed public method bodies with
owned CPU dummy buffers: nonzero FP32 byte offsets and release-before-return
are independently checked. No live runtime/private device memory exists in it.

Every D2H interval must be covered by completed H2D calls in the initialized
ledger. Read only the complete embedding shard or chosen W1/scale interval,
NEVER the uninitialized11.3GB W1 or adjacent slots. Wrong address/layout/bytes
reject. Readback proves these selected byte intervals and host/ABI addressing,
not every expert/rank/layer mapping, GEMM interpretation, full40 initialized
firsttouch or forward parity. No guard sentinel or implicit other weight copy.

## Planning budget and runtime guards

Base accepted empty reserve17,954,081,472B/rank is retained (N128 conservative
hostIO7,609,876,736 + nominalcomm680,528,320 +1GiB fixed planning +8GiB safety).
This operation additionally proposes128MiB/rank conservative transfer headroom:
TOTAL18,088,299,200B/rank. It is policy PLANNING, not measured device staging.
Public fallback copy staging is a HOST/shared buffer, bounded by64MiB/call;
this extra device headroom does not relabel host staging as measured device use.
Root must approve this policy hash for actual execution. All unknown full40
firsttouch/upload/forward peak fields remain null.

Before each full57 allocation and AFTER it, directfree>=remainingstatic+floor.
After full residency, before/after each copy require targetrank free>=floor;
all16 are checked before every selected source read. Do NOT subtract committed,
ring/prewarm again or multiply heap by168identitycapacity. Public totals/commit/
free, logical handles and copy byte counts are recorded. Prior last-minfree
20,251,516,928B is historical EMPTY observation, not future transfer capacity.

Proposed untouched fresh HOST/DISK/SHM planning floors64GiB/2GiB/16GiB, checked
before prepare/source reads and every copy. Parent RSS/PSS, global MemAvailable,
work-disk and SHM available are sampled; source raw+backing live point is sampled
before raw release. These are snapshots, not guaranteed between-call peaks or
whole-tree sums. Independent actual runner should retain the existing own-tree
PSS/USS/SHM monitor without summing fork RSS or modifying this sealed code.

Largest source raw+bytearray bound2,647,654,400B; after reader return the
1,323,827,200B backing survives under its330,956,800B view until4DP consumers
complete. One330,956,800B readback tensor and one64MiB public staging buffer are
live serially. Existing5,362,417,664B serial host allowance stays a planning
bound, not measuredRSS. Quant intermediate INT16/FP64/transpose/NZ temporaries
are within the selected11.8MB case, no whole40expert arrays. Source backing
weakrefs must be dead after consumer+readback return/del/gc BEFORE next group.

Free all registered handles, record after-free state, normalclose. On success,
prepare an EXTRA fresh owner with no weights; check prewarm AND after-free
queries against the transfer prewarm and accepted calibrationA at original
perrank tolerances, then normalclose. Success is pending until this final check.
Failure preserves first cause/cleanup errors, never continues source reads or
performs an automatic retry. Recorded PID/startticks/process snapshots still
need an independent post-job absence/resource audit; close is not a zero-leak
or full-peak proof.

## Separate future Root authorization and command

plan.json pins accepted actualcalib3bc729...27f0, probe c4b62...953e and external
PASS_EMPTY_RESERVATION_WITH_FINAL_AFTER_FREE_GATE, corrected1542, empty adapter
8d3 and real40 artifactreceiptc2cc. They are historical operation prerequisites,
not current free measurements, a new transfer peak or permission. Original
tolerances are used conservatively with fresh baseline queries. Their old empty
operation grant/expiry cannot authorize this new operation. Root may require a
fresh calibration separately; no missing measurement is filled with0.

Default CLI refuses. No valid actual authorization is included. Required JSON:

    schema: bounded-real-weight-transfer-authorization/1
    approved: true (ONLY separate explicit Root assignment)
    execution_kind: ACTUAL_DEVICE_BOUNDED_TRANSFER
    expires_at_epoch: actual future integer, <=3600s
    binding:
      operation: bounded-real-weight-transfer
      plan_sha256: frozen plan.json SHA
      code_sha256: frozen transport.py SHA
      policy_sha256: transport.digest(POLICY)
      source45_sha256: 762dc...
      device_ids: exact ACTUAL ordered16 TASK_DEVICE IDs
      byte_caps: exactly plan.byte_caps

Task-local auth creation after allocation is permissible only under explicit
RootGO, with actual TASK_DEVICE recorded, not assumed0..15 from compile. Exact
ordered IDs must match accepted calibration. First independent CPU/source gate:

    task-submit --no-device --max-time180 --run 'cd <REPO> && source activate-pypto.sh && python <OWN-FRESH-COPY>/check_cpu.py <fresh-output-name>'

Header metadata can be checked via make_plan.py --no-device; do not rewrite the
sealed plan. Tests use only owned tiny fake safetensors; no real tensor reads.

Only AFTER independent gate and separate actual-payload/NPU RootGO:

    task-submit --device auto --device-num 16 --max-time <assigned> --run 'cd <REPO> && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python <UNIT>/transport.py --execute --authorization <new-assigned-auth.json> --plan-file <UNIT>/plan.json --output <new-owned-output>'

Never run production device_run, remove fullweightguard, dispatch any model,
read additional keys/earlier leaves or whole uninitialized targets. STOP on
scope/header/range/dtype/cap/source/ABI/pin/IDs/memory/copy/hash/lifetime/query /
cleanup/finalbaseline discrepancy. Record partial completed transfers honestly;
failed API calls may have unknown partial effects and are not counted complete.
No retry/new key/larger heap/threshold relaxation/admin action. Next independent
acceptance must audit raw copy/byte/dtype/NZ hashes, coverage/lifetimes, all912
frees/normal closes/finalfresh before+after/PIDs and unchanged protected hashes.
Even success leaves full40 initialized upload/numerical forward/fit/text and the
production guard open/HELD.
