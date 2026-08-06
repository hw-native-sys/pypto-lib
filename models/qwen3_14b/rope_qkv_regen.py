# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # compile-only regeneration source; not a runnable kernel test
"""Regeneration source for ``kernels/paged_attention_cce/kernel/rope_qkv_generated.hpp``.

The fused ``paged_attention_rope_cce`` extern embeds a pypto/ptoas-GENERATED
``rope_qkv`` body as an in-kernel phase 0. That body is a *copied codegen
artifact*: it cannot be hand-edited safely, because ptoas CSEs constants by
value and the batch constants collide with unrelated ones (``16`` is both the
batch divisor and the Q-RMSNorm tile row count; ``128`` is both the item bound
``NUM_KV_HEADS * BATCH_PAD`` and ``HEAD_DIM``).

Commit ``8ef71cc`` (#796) folded the RoPE scope into the extern and DELETED the
pypto scope it was generated from, which left the header's "regenerate from
decode_fwd rope_qkv" instruction pointing at code that no longer exists. This
module is that missing source: a standalone, AIV-only program whose single
``pl.spmd(ROPE_CORES, name_hint="rope_qkv")`` scope reproduces the phase-0
arithmetic exactly, so the header can be regenerated on demand.

Regenerate with::

    python models/qwen3_14b/rope_qkv_regen.py -p a2a3

then follow the extraction steps printed at the end of the run.

Invariants this file must preserve (violating any of them silently changes the
generated GM addressing, which the extern cannot detect):

* **Column extents must match decode_fwd exactly** -- ``q_proj`` HIDDEN,
  ``k_proj``/``v_proj`` KV_HIDDEN, ``q_tnd_flat``/``k_cache``/``v_cache``/
  ``rope_cos``/``rope_sin``/``q_norm_w``/``k_norm_w`` HEAD_DIM. Every stride in
  the generated body is a baked literal derived from these.
* **Row counts do not matter.** They never enter the address math, which is why
  the internal buffers can stay padded at BATCH_PAD while the public batch is
  dynamic.
* **AIV-only.** A single matmul would split the program into aic/aiv halves and
  the artifact would no longer be a drop-in for the VEC-guarded call site.
* **``k_cache``/``v_cache`` stay statically shaped** -- no ``bind_dynamic``. The
  row count is provably dead in the generated body, and binding it would emit a
  second ``int64_t`` dynamic-dim parameter next to the batch one, making the two
  trivially swappable at the hand-written call site in ``fai_body.hpp``.
"""

import argparse
import re
from pathlib import Path

import pypto.language as pl
import torch
from pypto.backend import BackendType, set_backend_type
from pypto.runtime import RunConfig

from config import QWEN3_14B_DIMS as D, QWEN3_14B_TILING as T, QWEN3_14B as M

BATCH_PAD = M.batch_pad
BATCH_DYN = D.batch
NUM_HEADS = M.num_heads
NUM_KV_HEADS = M.num_kv_heads
HEAD_DIM = M.head_dim
HALF_DIM = M.half_dim
HIDDEN = M.hidden
KV_HIDDEN = M.kv_hidden
Q_PER_KV = M.q_per_kv
Q_HEAD_PAD = M.q_head_pad
HEAD_DIM_INV = M.head_dim_inv
EPS = M.eps
MAX_SEQ = M.max_seq
BLOCK_SIZE = T.block_size

# Must stay identical to decode_fwd.py's values -- tests/contract cross-checks
# ROPE_CORES against kQwenRopeCores in fai_body.hpp and the provenance banner in
# the generated header.
ROPE_CORES = 32
# Upper bound at the PADDED batch. The trip count stays a compile-time constant
# so the pipeline keeps its fixed two-guarded-blocks-per-iteration shape; the
# `g_idx < NUM_KV_HEADS * batch` guard masks the tail at smaller runtime batches.
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH_PAD + ROPE_CORES - 1) // ROPE_CORES

# Sized like decode_fwd's standalone paged pool. Only the COLUMN count matters.
DECODE_MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
CACHE_ROWS = BATCH_PAD * DECODE_MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE

K_RED_ROWS = 8

# A minimal per-layer loop purely so layer_cache_base is an opaque runtime
# scalar in the emitted kernel (see the comment at its use site).
_LAYER_LOOP_TRIPS = 2
_LAYER_CACHE_ROWS = CACHE_ROWS // _LAYER_LOOP_TRIPS

# Flip to make the batch a runtime value. Kept as a switch (rather than deleted)
# so the fidelity gate can regenerate with the batch pinned to BATCH_PAD and
# diff the body against the shipped header before any dynamism is introduced.
DYNAMIC_BATCH = True

# Resolved OUTSIDE the traced function on purpose. Inside a `@pl.jit` body the
# frontend rewrites a plain `if` into IR control flow (which would make `batch`
# scope-local and fail SSA verification) and rejects calls to ordinary Python
# helpers, so the switch can only be expressed in the type annotation.
_SEQ_DIM = BATCH_DYN if DYNAMIC_BATCH else BATCH_PAD


@pl.jit
def rope_qkv_regen(  # noqa: PLR0913 -- mirrors the extern's packed arg list
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16]],
    q_tnd_flat: pl.Out[pl.Tensor[[BATCH_PAD * NUM_HEADS, HEAD_DIM], pl.BF16]],
    seq_lens: pl.Tensor[[_SEQ_DIM], pl.INT32],
    inv_rms_states: pl.Tensor[[BATCH_PAD, 1], pl.FP32],
    slot_mapping: pl.Tensor[[BATCH_PAD], pl.INT32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    q_proj: pl.Tensor[[BATCH_PAD, HIDDEN], pl.FP32],
    k_proj: pl.Tensor[[BATCH_PAD, KV_HIDDEN], pl.FP32],
    v_proj: pl.Tensor[[BATCH_PAD, KV_HIDDEN], pl.FP32],
    q_norm_w: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_w: pl.Tensor[[1, HEAD_DIM], pl.FP32],
):
    # The batch comes from the seq_lens annotation alone -- no bind_dynamic call.
    # The frontend rejects both plain `if` statements (traced as IR control flow)
    # and arbitrary Python helper calls inside a jit body, so the DYNAMIC_BATCH
    # switch has to live entirely in _SEQ_DIM. With a static _SEQ_DIM this folds
    # to the BATCH_PAD constant at trace time, which is what the fidelity gate
    # compares against the shipped header.
    batch = pl.tensor.dim(seq_lens, 0)

    # layer_cache_base must reach the generated body as an OPAQUE runtime scalar,
    # because the extern supplies its own cache_row_offset in that parameter slot.
    # An entry `pl.Scalar` does NOT work: lower specializes it on the
    # traced value and the parameter goes dead (`0 + x` folds; a non-zero probe is
    # baked as a literal). Deriving it from a pl.range induction variable
    # reproduces how decode_fwd's per-layer inline loop produced the shipped ABI.
    # Stride the loop so the induction variable IS the base: a separate
    # `layer_idx` would be captured by the scope as a second, dead scalar arg.
    for layer_cache_base in pl.range(0, CACHE_ROWS, _LAYER_CACHE_ROWS):

        # QK-norm and RoPE retain the main implementation's fused arithmetic,
        # but run as a standalone producer for the external CANN attention.
        with pl.spmd(ROPE_CORES, name_hint="rope_qkv") as rope_tid:
            rope_core = pl.get_block_idx()
            q_red_pad = pl.full(
                [1, (Q_HEAD_PAD - Q_PER_KV) * HEAD_DIM],
                dtype=pl.FP32,
                value=0.0,
            )
            k_red_pad = pl.full(
                [1, (K_RED_ROWS - 1) * HEAD_DIM],
                dtype=pl.FP32,
                value=0.0,
            )
            for it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
                g_idx = rope_core + it * ROPE_CORES
                if g_idx < NUM_KV_HEADS * batch:
                    ki = g_idx // batch
                    b = g_idx - ki * batch
                    ctx_len = pl.read(seq_lens, [b])
                    inv_rms_b = pl.read(inv_rms_states, [b, 0])
                    pos = ctx_len - 1
                    wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
                    wr_slot_block = wr_slot // BLOCK_SIZE
                    wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
                    cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
                    cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
                    sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
                    sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

                    kv_col = ki * HEAD_DIM
                    k_raw = pl.mul(
                        pl.reshape(
                            pl.concat(
                                k_proj[b : b + 1, kv_col : kv_col + HEAD_DIM],
                                k_red_pad,
                            ),
                            [K_RED_ROWS, HEAD_DIM],
                        ),
                        inv_rms_b,
                    )
                    k_ss = pl.row_sum(pl.mul(k_raw, k_raw))
                    k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                    k_normed = pl.row_expand_mul(
                        pl.col_expand_mul(k_raw, k_norm_w),
                        k_inv,
                    )
                    k_full = k_normed[0:1, :]
                    k_lo = k_full[:, 0:HALF_DIM]
                    k_hi = k_full[:, HALF_DIM:HEAD_DIM]
                    rot_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    rot_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    cache_row = (
                        layer_cache_base
                        + (wr_slot_block * BLOCK_SIZE + wr_slot_offset) * NUM_KV_HEADS
                        + ki
                    )
                    k_cache = pl.assemble(
                        k_cache,
                        pl.cast(pl.concat(rot_lo, rot_hi), target_type=pl.BF16),
                        [cache_row, 0],
                    )
                    v_row_bf16 = pl.cast(
                        pl.mul(
                            v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM],
                            inv_rms_b,
                        ),
                        target_type=pl.BF16,
                    )
                    v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                    q_base = ki * Q_PER_KV
                    q_raw = pl.mul(
                        pl.reshape(
                            pl.concat(
                                q_proj[
                                    b : b + 1,
                                    q_base * HEAD_DIM : (q_base + Q_PER_KV) * HEAD_DIM,
                                ],
                                q_red_pad,
                            ),
                            [Q_HEAD_PAD, HEAD_DIM],
                        ),
                        inv_rms_b,
                    )
                    q_ss = pl.row_sum(pl.mul(q_raw, q_raw))
                    q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                    q_heads = pl.row_expand_mul(
                        pl.col_expand_mul(q_raw, q_norm_w),
                        q_inv,
                    )
                    q_lo = q_heads[:, 0:HALF_DIM]
                    q_hi = q_heads[:, HALF_DIM:HEAD_DIM]
                    q_rot_lo = pl.sub(
                        pl.col_expand_mul(q_lo, cos_lo),
                        pl.col_expand_mul(q_hi, sin_lo),
                    )
                    q_rot_hi = pl.add(
                        pl.col_expand_mul(q_hi, cos_hi),
                        pl.col_expand_mul(q_lo, sin_hi),
                    )
                    q_row = b * NUM_HEADS + q_base
                    q_tnd_flat = pl.assemble(
                        q_tnd_flat,
                        pl.cast(
                            pl.concat(q_rot_lo, q_rot_hi)[0:Q_PER_KV, :],
                            target_type=pl.BF16,
                        ),
                        [q_row, 0],
                    )
    return q_tnd_flat


def _dummy_inputs(batch: int):
    """Shapes only -- this program is compiled, never run.

    Only `seq_lens` is sized from `batch`; every other buffer stays padded at
    BATCH_PAD, and the traced scope indexes them by the batch component of the
    item id. A batch above BATCH_PAD would therefore trace reads past those
    buffers, so reject it here where the failure is still legible.
    """
    if not 1 <= batch <= BATCH_PAD:
        raise ValueError(
            f"batch must be in [1, BATCH_PAD={BATCH_PAD}], got {batch}; the "
            "padded buffers cannot address more rows than BATCH_PAD"
        )
    return (
        torch.zeros([CACHE_ROWS, HEAD_DIM], dtype=torch.bfloat16),
        torch.zeros([CACHE_ROWS, HEAD_DIM], dtype=torch.bfloat16),
        torch.zeros([BATCH_PAD * NUM_HEADS, HEAD_DIM], dtype=torch.bfloat16),
        torch.ones([batch], dtype=torch.int32),
        torch.ones([BATCH_PAD, 1], dtype=torch.float32),
        torch.zeros([BATCH_PAD], dtype=torch.int32),
        torch.zeros([MAX_SEQ, HEAD_DIM], dtype=torch.float32),
        torch.zeros([MAX_SEQ, HEAD_DIM], dtype=torch.float32),
        torch.zeros([BATCH_PAD, HIDDEN], dtype=torch.float32),
        torch.zeros([BATCH_PAD, KV_HIDDEN], dtype=torch.float32),
        torch.zeros([BATCH_PAD, KV_HIDDEN], dtype=torch.float32),
        torch.ones([1, HEAD_DIM], dtype=torch.float32),
        torch.ones([1, HEAD_DIM], dtype=torch.float32),
    )


def _newest_artifact() -> Path:
    out_root = Path(__file__).parent / "build_output"
    builds = sorted(out_root.glob("_jit_rope_qkv_regen_*"), key=lambda p: p.stat().st_mtime)
    if not builds:
        raise SystemExit("no _jit_rope_qkv_regen_* build found -- compile first")
    return builds[-1] / "kernels" / "aiv" / "rope_qkv.cpp"


def _emit_header(artifact: Path, dst: Path) -> None:
    """Re-wrap a generated rope_qkv.cpp as the embeddable header.

    Keeps the ptoas preamble helpers and the rope_qkv body verbatim, drops the
    kernel_entry launcher (fai_body.hpp transcribes its arg unpacking by hand),
    and emits the mechanical parameter table the mapping there must mirror.

    Writes to `dst`, which deliberately defaults to a STAGING path rather than
    the live header: the shipped header carries hand-maintained prose (the
    semantic `meaning` column, the DO-NOT-EDIT rationale) that this cannot
    regenerate. Diff the two and merge the body + parameter table across.
    """
    src = artifact.read_text()
    start = src.index("static __aicore__ void rope_qkv(")
    depth, idx = 0, src.index("{", start)
    while True:
        if src[idx] == "{":
            depth += 1
        elif src[idx] == "}":
            depth -= 1
            if depth == 0:
                break
        idx += 1
    body = src[start:idx + 1]
    marker = "// --- ptoas-generated code ---"
    preamble = src[src.index(marker) + len(marker):start].strip("\n")

    launcher = src[src.index('extern "C"'):]
    names = [m.split(": ", 1)[1] for m in
             re.findall(r"// (?:Unpack \w+|Extract dynamic dim): .*", launcher)]
    call = re.search(r"rope_qkv\((.*?)\);", launcher, re.DOTALL).group(1)
    call_args = [a.strip() for a in call.split(",")]
    # Parameter names come from the SIGNATURE line only -- the body redeclares
    # plenty of vNN temporaries that would otherwise be picked up.
    sig = body[:body.index(") {") + 1]
    params = re.findall(r"\b(v\d+)\s*[,)]", sig)

    table = "\n".join(
        "//   %2d.  %-5s %s" % (n, params[n] if n < len(params) else "??", a)
        for n, a in enumerate(call_args))

    dst.write_text(
        "// GENERATED PARAMETER ORDER (mechanical -- regenerate, do not hand-edit):\n"
        "// pypto draws lifted scalars from the same name pool as dynamic dims, so\n"
        "// an emitted name like BATCH_DYN may NOT be the batch. Trust positions.\n"
        f"{table}\n"
        "#ifndef PYPTO_QWEN_ROPE_QKV_GENERATED_HPP\n"
        "#define PYPTO_QWEN_ROPE_QKV_GENERATED_HPP\n\n"
        "#include <cstdint>\n\n"
        "#ifdef __DAV_C220_VEC__\n"
        "#include <pto/pto-inst.hpp>\n"
        '#include "tensor.h"\n'
        '#include "intrinsic.h"\n\n'
        "namespace qwen_rope_gen {\n"
        "using namespace pto;\n"
        f"{preamble}\n\n{body}\n\n"
        "}  // namespace qwen_rope_gen\n"
        "#endif  // __DAV_C220_VEC__\n\n"
        "#endif  // PYPTO_QWEN_ROPE_QKV_GENERATED_HPP\n")
    print(f"wrote staging header: {dst}")
    print("\nparameter order (fai_body.hpp must mirror these POSITIONS):")
    print(table)
    print("\nnext: diff against kernels/paged_attention_cce/kernel/"
          "rope_qkv_generated.hpp and merge the body + table across, keeping\n"
          "      that file's hand-written prose and `// ROPE_CORES:` banner.")


def _backend_type(platform: str) -> BackendType:
    # The fused RoPE+FAI extern this artifact feeds is A2/A3-only, matching
    # decode_fwd._backend_type / paged_attention_cce.SUPPORTED_PLATFORMS.
    return BackendType.Ascend910B


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=("a2a3", "a2a3sim"))
    # CI runs every changed model file as `python <file> -p <platform> -d <device>`,
    # so the device flag must parse even though this program is compile-only and
    # never reaches a device.
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="accepted for CI-harness uniformity; compile-only, unused.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH_PAD,
                        help="batch used for the dummy seq_lens/slot_mapping shapes; with "
                             "DYNAMIC_BATCH the compiled body is batch-independent, so this "
                             "only affects the traced example shape.")
    parser.add_argument("--emit-header", nargs="?", const="rope_qkv_generated.staged.hpp",
                        default=None, metavar="PATH",
                        help="after compiling, re-wrap the artifact into an embeddable "
                             "header at PATH (default: ./rope_qkv_generated.staged.hpp). "
                             "Staging on purpose -- the live header has hand-written prose "
                             "to preserve; diff and merge.")
    args = parser.parse_args()

    set_backend_type(_backend_type(args.platform))
    # compile(), NOT lower(): this module exists to produce the codegen artifact
    # that --emit-header re-wraps (_newest_artifact reads
    # build_output/_jit_rope_qkv_regen_*/kernels/aiv/rope_qkv.cpp). lower() is
    # codegen-free and artifact-free by contract, so it would leave the header
    # regeneration with nothing to read -- and silently, since the run itself
    # succeeds and CI never passes --emit-header.
    #
    # The platform goes in explicitly so compile() cannot fall back to the
    # DEFAULT backend and collide with the set_backend_type above.
    compiled = rope_qkv_regen.compile(
        *_dummy_inputs(args.batch),
        config=RunConfig(platform=args.platform, backend_type=_backend_type(args.platform)),
    )
    post_pass = compiled.program
    print(f"Compiled program has {len(post_pass.functions)} function(s):")
    for fn in post_pass.functions.values():
        print(f"  {fn.name}: {fn.func_type}")

    if args.emit_header:
        _emit_header(_newest_artifact(), Path(args.emit_header))
        raise SystemExit(0)

    out_root = Path(__file__).parent / "build_output"
    builds = sorted(out_root.glob("_jit_rope_qkv_regen_*"), key=lambda p: p.stat().st_mtime)
    if builds:
        artifact = builds[-1] / "kernels" / "aiv" / "rope_qkv.cpp"
        print(f"\nDYNAMIC_BATCH={DYNAMIC_BATCH}  ROPE_ITEMS_PER_CORE={ROPE_ITEMS_PER_CORE}")
        print(f"generated artifact: {artifact}")
        print(
            "\nTo refresh kernels/paged_attention_cce/kernel/rope_qkv_generated.hpp:\n"
            "  1. keep the `static __aicore__ void rope_qkv(` function body and the\n"
            "     ptoas preamble helpers; DROP the trailing `kernel_entry` block\n"
            "  2. re-wrap: license header + provenance banners (incl. `// ROPE_CORES:`)\n"
            "     + #ifndef PYPTO_QWEN_ROPE_QKV_GENERATED_HPP + #ifdef __DAV_C220_VEC__\n"
            "     + namespace qwen_rope_gen { ... }\n"
            "  3. paste the launcher's `// Unpack tensor:` list and its final forwarding\n"
            "     call into the header preamble as a comment -- fai_body.hpp's hand-written\n"
            "     arg mapping is derived from it\n"
            "  4. update fai_body.hpp's call + its static_assert on the function-pointer type"
        )
