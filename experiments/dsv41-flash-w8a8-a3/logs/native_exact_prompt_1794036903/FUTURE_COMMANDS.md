# Exact three-token validation proposal

This document requests observer review. It grants no native or forward execution.

The accepted CPU work produces a new N3 recipe and 57 hashes. It does not reuse
the N128 hashes. The input is exactly `[100,101,102]`, positions `[0,1,2]`, all
sixteen `token_counts=3`, and sampling row 2. This synthetic fixture uses forty
dense layers and one reused expert layer, with Engram disabled.

Before a future stage, create a fresh owned capture directory under these logs.
Copy `sealed_main`, the 45-file `producer_source`, `derived_model`,
`reference_n3.py`, `case_n3.py`, `n3_stages.py`, `reference_policy.json`,
`binding_results.json`, and both JSON files in `n3_work` byte-for-byte. Record
lineage to this unit's final manifest and static plan SHA. No tensor payload
exists in the current static work. Leave this unit byte-immutable. Relative
module paths and the recipe permit this fork before the first capture; compiled
capsules created later bind absolute copied ABI paths, so prefix/offline/decode
then stay in that new physical work.

The commands below assume `CAPTURE` names the new owned directory and the
observer has issued the specific stage GO. Authorization JSON must encode that
directive, exact current file/dependency/static plan/policy SHA values, and a
fresh actual `/proc/meminfo`/`statvfs` measurement. The helper rejects omitted,
wrong-stage, changed-source, stale, undersized, or fake measurement scope.
The static-only authorization cannot unlock any command below.

```bash
task-submit --device auto --device-num 16 --max-time 1800 \
  'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYPTO_PROG_BUILD_DIR=CAPTURE/build python CAPTURE/n3_stages.py --stage prefix --execute --authorization CAPTURE/prefix_root_go.json'
```

The native prefix command performs no reference computation. It uses the NEW
generate module for factory, configuration, resident owner, and dispatch. It
extracts only the six recorded setup/telemetry helper definitions from pinned
old evidence; it never imports or runs the old helper's generate module. Its
public configuration is normalized ONBOARD, a2a3, ring `[0,0,4294967296,0]` for
prepare and every call. The pinned installed TMR PROGRAM contract must support
shared arena bank 0; directory capacity is not bank count. Per-rank post-prewarm
free memory must exceed 23,344,218,656 B and post-upload free must exceed 8 GiB.
The original checks remain fail closed.

After native terminal/copyback, `n3_work/prefix3` atomically contains actual and
independent pristine initial nonstatic bytes, copied compiled ABI, source/recipe/
57 static hashes, original artifact/owner/config identity, and a normal-close
marker. No CPU reference occurs before persistence. No native-only numerical
claim is valid. Verify the job has ended and its own workers/locks are absent
before a separate CPU stage:

```bash
task-submit --no-device --max-time 7200 \
  'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python CAPTURE/n3_stages.py --stage validate --execute --authorization CAPTURE/validate_root_go.json'
```

Offline validation reconstructs/checks N3 static bytes and the full concrete
100-argument ABI. It requires the unchanged accepted parameter directions and
layouts, including cache InOuts. It runs original unconditioned followed by
actual-snapshot staged references and all original 17 owning gates, with exact
16 prefix mutables initialized inside the same isolated reference context,
pristine hashes around each pass and static hashes before/after. Direct pool/
count gates and comparison tolerances are unchanged. CPU evaluation uses only
the four previously proven exact bounded integer-dot substitutions. It never
initializes native preparation or reads real checkpoint tensors.

Only after observer acceptance of that prefix17 receipt, a distinct GO may run
ONE actual decode at position 3, using the saved actual prefix bytes:

```bash
task-submit --device auto --device-num 16 --max-time 1800 \
  'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYPTO_PROG_BUILD_DIR=CAPTURE/build python CAPTURE/n3_stages.py --stage decode --execute --authorization CAPTURE/decode_root_go.json'
task-submit --no-device --max-time 7200 \
  'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python CAPTURE/n3_stages.py --stage validate-decode --execute --authorization CAPTURE/decode_cpu_root_go.json'
```

The decode input token is fixed 17 for this numerical case. This is not text
generation. Source position 2 leaves an unfinished ratio-2 C2 pair; actual
decode position 3 must consume the returned partial `state_cache_c2a` unchanged
and publish logical/physical compressed row 1. The handoff records its hash and
parent manifest before dispatch. Original decode19 gates then run both
references with exact 12 mutable names and pristine/static checks. No decode
job is authorized by the current CPU unit.

Recorded AND fresh byte budgets (conservative N128 source floors plus margins):

| Stage | HOST | additional DISK | SHM |
| --- | ---: | ---: | ---: |
| prefix | 290783361536 | 23876796928 | 37231460864 |
| offline prefix | 306003115008 | 8657043456 | 8589934592 |
| decode position3 | 290783361536 | 10633936896 | 37231460864 |
| offline decode | 306003115008 | 8657043456 | 8589934592 |

These are guards, not measured real-weight fit. Full weights and production
generation remain held. Any new native/serialization/close error stops the
case. A recoverable CPU failure is diagnosed from saved capsules, without a
blind native rerun.
