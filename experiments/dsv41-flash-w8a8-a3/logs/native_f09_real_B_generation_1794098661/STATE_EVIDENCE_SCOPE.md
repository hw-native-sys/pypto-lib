# Available state diagnostics and limits

The saved generation.json contains state fields on each of15 decode steps:
window_cache_sha1, compressed_cache_c2a_sha1, index_cache_c2a_sha1,
state_cache_c2a_sha1, compressed_cache_c1a_sha1, index_cache_c1a_sha1,
x_hc_sha1, pre_mix_sha1, topk_indices_nonzero, pool_block_ids_nonzero,
and position_ids. The owner initially searched only sha256/digest and incorrectly
reported no state hashes; that statement was retracted explicitly to Root and
the independent auditor before closure. Original output bytes were not edited.

Frozen generate.py223-225 computes each diagnostic by converting the tensor to
contiguous float32, taking numpy bytes, and recording SHA1 hexdigest[:16].
These are truncated fingerprints of float32-converted returned host values,
not cryptographic hashes of the original cache-storage bytes. The result file
SHA256 separately binds serialized metadata and must not be called a tensor
payload hash. No raw cache tensor payload was reread/saved/recomputed by this
owner metadata review; full cache numerical/original parity remains unproven.

Saved first decode at position3: window f809282657816483,
compressedC2 5d2753e718ed2476, stateC2 ee9deb4639bef9d4.
Saved final decode step15 at position17: window b65211cc901b9fcd,
compressedC2 dd74c65d1afb16c2, indexC2 db1b149b01c5746a,
stateC2 34eda82525798c6a, compressedC1 99bc29f6945e52e1,
indexC1 85df88f8ad46a7de, x_hc ae85eea8c8eeb48b,
pre_mix 1c7218c4746ade2e, topk nonzero1872/pool nonzero144.
Presence, sequencing and source/ABI binding are available for audit; diagnostic
changes alone are not a mathematical correctness or byte-parity proof.

Existing postqueue snapshot exact first lines:
`=== 设备占用 (0/16) ===`
`  全部空闲`
Pending and Running sections are empty. Earlier owner wording "0/16 devicesfree"
was incorrect; it was corrected to the exact occupancy line. Actual release
proof separately retains raw release0..15 and known birth absence. No new
queue query/test was required for that wording correction.
