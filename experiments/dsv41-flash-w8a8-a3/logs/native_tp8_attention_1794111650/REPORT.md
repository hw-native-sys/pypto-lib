# TP8 eight-head attention native validation

PASS: ONE auto1 task task_20261108_122101_13767445722 completed exit0 on actual allocated card0. Original core16, packed640 and window128 each passed the original output comparator for BF16 (3,8,512), atol=.02,rtol=.02,max_error_ratio=.005. This validates eight live heads with sixteen physical Cube rows, partial visible key tiles and random per-head sink values. No compute entry, golden or threshold was changed.

Frozen original CLI ran through wrapper73f049 using attention562cda/configabfb. --sim was absent, its original CLI provides execution_mode=False and the pinned normalize_config maps this to ONBOARD. All three generated compiled_meta.json explicitly show a2a3 and logical eight-head shapes. Each case has actual native Callable image/run/device_wall evidence, followed by callable-image free. CLI returned successfully only after all original RunResult.passed checks. Task raw log reports normal device0 lock release; fresh queue snapshot follows terminal.

Frozen preparation53 and all pinned source/compiler/harness/library dependencies remain byte-identical. No source, old evidence, checkpoint payload, model weights, full model/16-card run, sampling or generation was changed/performed. The planning floors were checked against actual prequeue host/disk/SHM capacity; no runtime memory peak is claimed.

Cleanup scope: logged runtime/lock PIDs are absent now. The first process observation happened after this short job had already completed, so no original PID birth ticks were captured; known-birth identity list is honestly empty. Numeric gates, native terminal and raw lock release are independently preserved. This does not imply full TP8 model correctness, arbitrary nonmultiple16 physical key lengths, all-masked native coverage, F09 closure, or real-weight text quality.

No retries; FINAL/PARK.
