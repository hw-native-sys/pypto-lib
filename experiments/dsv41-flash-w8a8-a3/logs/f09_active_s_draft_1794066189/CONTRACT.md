# Draft scalar / capacity contract, before dispatch

UNAPPLIED only. `moe_ffn(..., x_hc_out, next_pre_mix, my_rank, epoch, num_tokens: pl.Scalar[pl.INT32], rows: pl.constexpr)`.
Existing `rows` remains compile-time receive/return capacity. Physical token capacity2 in targetcase. Actual `num_tokens` is independently supplied runtime prefixcount0..capacity from token_counts[rank]; forward callers pass this after epoch and before keywordrows. No clamping invalid values tohide contract errors. All16EP ranks participate/servepeers evenlocalactive0. Dynamic `moe_ep16` call uses capacity=rows,rows=0 and views ofactualactive length. Keep full-capacity scratch backing for possibleempty views / anchorreads; do not allocatezero-backed route scratch. HC/gate/shared+route/combine/post meaningful writes onlyactive prefix. Do NOTchange F10stride/layout/counters/math/tols.

Operatorgolden optional tensors['active_counts'] defaultsallphysicalT; independently validated per-sourcecount controls routed/shared expertinput/output. Inactive routedoutput/carry unchanged; destinationexperts stillserveother sources. No expectedmaskfromcandidate. No compile/nativeproof inthisunit.

Main uses token_counts through layer/fullreference and calls. Fullcapacitycaller scratchtails can beundefined; persistentidlecache/state MUSTbe preserved. Standaloneunsupported staged-active layerrefs mustfailclosed ifnotimplemented; fullentry reference path is thetarget.

Targetschema explicit: physical_token_capacity2, request_capacity2, logical_pages_per_request2, physical_pages4 (A[3,1],B[0,2]), state_blocks2 (A1,B0). No ambiguous pages2 field.

WorkerexclusiveONLYNEWoperator/**; nevereditRootCCsource orMainproposedfiles/ledgers. AfterworkerFINAL/PARK Mainmaycopy finaloperatorfile intoMainproposed/ as integrationsnapshot; originalworkerfile staysimmutable.
