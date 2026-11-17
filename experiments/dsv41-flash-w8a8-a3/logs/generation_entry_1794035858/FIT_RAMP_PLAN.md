# Current-source empty-fit route — prepared only, hardware guard remains

Existing real_fit_readiness_1794018846 header-only budget and
fit_probe_prepare_1794020737/fit_probe.py are unchanged. This unit's fit_probe.py
is a copy with ONLY before.json->bound_source.json metadata binding and explicit
ring[0,0,4294967296,0] requirement; fit_probe_rebind.patch records that exact diff.
No allocator/runtime algorithm or public API changed. bound_source.json equals
new producer45 after.json762dc...; default execution still refuses. CPU metadata
checks confirm57resources/39,624,816,576B perrank and max11,324,620,800B, both
phase layouts, current source and missing/2GiB/3GiB ring rejection. No owner,
large allocation, payload read or NPU ran here.

Future root-assigned reviewer sequence:

1. Check new source/diff/config and independent odd-N functional disposition;
   use a real40/57-resource signature-only a2a3 artifact, not E1 fixture metadata.
   No tensor samples/global dummy stacks. Explicit generation public builder
   and normalized ONBOARD[0,0,4GiB,0] must be used for prewarm. Record artifact
   metadata+all57 rankshape/dtype bytes and installed selected TMR shared-GM
   contract/library24434f...; unknown backend/library/pin/query fails closed.
2. --no-device plan command, allowed before later hardware assignment:

       source activate-pypto.sh
       python <UNIT>/fit_probe.py --plan --phase prefill
       python <UNIT>/fit_probe.py --plan --phase decode

3. ONLY with separate rootGO, accepted functional/memory receipts and fresh
   public budget on all16 allocated devices, one empty-allocation job:

       task-submit --device auto --device-num 16 --max-time <assigned> --run
       'cd <REPO> && source activate-pypto.sh && python <UNIT>/fit_probe.py
       --execute --phase prefill --authorization <assigned-auth.json>
       --measurements <actual-calibration.json> --artifact <new-real40-artifact>
       --output <fresh-probe-output>'

   Replace UNIT/REPO with real absolute paths. Authorization/calibration bind
   NEW762dc source, current probe source/artifact metadata and independent
   reports. Calibration must carry exact4GiB ring vector, selected backend
   pins/evidence and16 MEASURED additional-peak, explicit safety-margin and
   post-close-baseline tolerance byte entries. No real calibration or allocation
   authorization is supplied here. Existing tiny/old E1 samples are not full-fit
   measurements; absent/unknown additional peak keeps the guard closed.

4. First owner: public prepare, perrank committed/free/total baseline. Allocate
   one largest11,324,620,800B empty tensor on EACH rank with alloc_tensor(init=None).
   Record each handle/schema/query. Free every handle, capture after-free state
   and CLOSE owner completely. free_tensor alone may retain allocator cache.
5. Fresh owner: same supported public config, requery baseline and reject
   regression beyond assigned measured tolerance. Ramp all57 resources perrank,
   total39,624,816,576B/rank,16ranks (912handles), querying before/after each.
   Free all handles/CLOSE. Largest and aggregate allocations never coexist.
   Optionally repeat decode phase only under its separately measured/auth scope.
   No forward, source weight payload, upload or uninitialized compute occurs.

Per-rank free guard is direct public free >= remaining target static + phaseIO
planning reserve + nominal comm + measured extra peak excluding those/already
committed + explicit margin. Do NOT subtract committed from free again; prewarm
ring is already charged. Existing helper deliberately reserves the full hostIO
payload perrank as a conservative planning allowance (prefill7,609,876,736B,
decode247,111,680B); it is not a measured device footprint or inferred sharding.
Nominal comm is680,528,320B prefill /4,663,744B decode perrank. A4GiB target ring2
plus3defaults256MiB is4.75GiB per shared bank/rank; actual public committed/free,
not168identity capacity, determines remaining budget. No extra bank multiplier
or doublecharge.4GiB is supported by actual public RunConfig power2 validation.

Prior independent E1 decode prewarm observed committed5,452,987,693B/rank and
minfree59,886,047,232B; these are comparison observations, NOT new real40-fit
baselines or guaranteed peaks. Physical source57static perrank39.625GB vs
bounded6.464GB has delta33,160,937,472B. Existing serial host source allowance
max(4*S,24*Q)+64MiB=5,362,417,664B remains a planning overbound, not measured
RSS; largest BF16 source is1,323,827,200B physical, raw+bytearray2,647,654,400B.
Real hostIO, transform/serialization, allocator/HCCL/VMM/transient peaks and
sequential owner cleanup still require measurement. No124GB replicated allocation
or full real checkpoint materialization is permitted by this route.

STOP on unknown/pin-changed backend, unsupported ring, missing source/functional/
measurement evidence, public query error, any rank insufficient free/remaining
headroom, contiguous allocation/schema failure, unexpected owner reuse, baseline
regression, cleanup error or lost owner. Preserve first error and own cleanup;
no larger-ring retry, admin/reset, guard relaxation or other-job intervention.
Empty-allocation success proves reservation/contiguous-address feasibility only;
it does not prove physical first-touch/upload peak, numeric fit, generation,
coherent text or permission to remove the hard full-weight guard.
