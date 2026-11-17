# Exact prepared identity contract (new helper only)

Installed source pins (also enforced by helper): device_runner489be949…d3e6,
_external_source7778f81b…b9b6, public elf_parser6c99eb36…1736. No private runtime
function is called. Source inspection at device_runner112–135/335–341 and
_external_source83–135 establishes this restricted generated contract:

    contract=(registered_code_name, source_filename, registered_core_type, ())
    digest = SHA256(framed('metadata',repr(contract)) + framed('name',source_filename)
                    + framed('content',exact_source_bytes))[:24]
    framed(label,payload)=label.encode()+len(payload).to_bytes(8,'little')+payload
    object=source_parent/(exact_stem+'_'+digest+'.o')
    cache=chip/cache/('incore_'+digest+'_'+core+'_'+exact_stem+'_a2a3.bin')

Only literal generated registrations with code_name, nonnegative int func_id,
core_type aic/aiv and empty extra_include_dirs are accepted. No existing quoted
source-local include is supported in this restricted helper; such a dependency
would change the pinned runtime digest closure and MUST STOP. External/builtin/
vpto/unknown registrations still STOP. Compiler-runtime headers searched internally
by the compiler are outside this empty-tuple naming digest; their toolchain context
remains separately Root-pinned. This receipt is not a recompile/source semantic proof.

All registered functions are parsed from unchanged source-bound kernel_config AST.
Exact source identity determines object and cache, never a glob first match. Entire
.o/.bin inventories in registered directories must match exactly. Thus sibling
base/_0/_1 stems are separate functions; malformed/extra/unregistered/cross-identity
images do not become a fallback match. Same stem with a wrong24hex is missing-own
identity, not an ambiguous selector to bypass.

For each actual object the pinned PUBLIC extract_text_section function is loaded
from elf_parser.py (pure CPU, no runtime initialization). Expected cache bytes must
equal that object's nonempty .text exactly, and both bytes/digests are recorded.
Orchestration ELF remains separately required. After successful future public prepare,
write a NEW owned artifact_after_prepare.json with all registered identities, source
projection and naming pins. verify_prepared_receipt rechecks them on CPU validation.

inspect_prepared is read-only and may inspect RETAINED actual failed output; it never
writes old receipts. after_public_prepare is used only on a NEW future owned prepare
(or explicitly labelled CPU-copy regression), never on the frozen failed case.
Original compile origins remain immutable/source-only. No preprepare .bin requirement.

Historical helper1cc9 used stem+'_*.o'. Retained actual32 registrations have four
colliding prefixes: int8_quant_per_token2, swiglu2, moe_compact2, moe_return4.
Actual task051030 FAILED before static upload/count calls despite successfully
prepared images. It remains FAILED; cleanup historical birth startticks UNKNOWN,
all16 present-PIDs absent/raw16 release facts do not repair missing historical data.
Old accepted tiny32768 proof only covers its unambiguous layout, not this new helper.
New helper acceptance requires independent CPU verification of retained actual images,
then NEW Root wrapper with NEW helper/route SHA; no old90d/1cc authorization relabel.

Before fullentry future native readiness: inspect all frozen fullentry registrations
(source/core/include layout) and compute expected naming identities, but do NOT claim
fullentry prepared/native proof until that future native stage actually does so.
Three signatures/source405/probes/oraclee45/model math are untouched.
