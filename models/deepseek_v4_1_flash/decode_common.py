# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mode-independent decode Attention half-layer boundaries and validation helpers."""

import argparse
from dataclasses import replace
from types import SimpleNamespace

from pathlib import Path

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import D, HC_MULT
from models.deepseek_v4_1_flash.golden import rms_norm as golden_rms_norm
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.decode_sp_integration import combine_validation, sequence_parallel_bounds
from models.deepseek_v4_1_flash.rmsnorm import rms_norm
from pypto.ir import DistributedConfig


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    for row in pl.spmd(tokens, name_hint="decode_sp_zero_padding"):
        if row >= valid_rows:
            x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


BOUNDARY_PREFIX_NAMES = (
    "x_hc",
    "incoming_pre_mix",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_weight",
)
BOUNDARY_OUTPUT_NAMES = (
    "attention_output",
    "attention_hidden",
    "attention_pre_mix",
)
SCALAR_NAMES = ("num_tokens", "attention_epoch")


@pl.jit.inline
def slab_owner(
    tp_rank: pl.Scalar[pl.INT32],
    slab: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Return ``(first, count)``: the active rows the physical slab *tp_rank* owns.

    ``slab`` is the rank's fixed physical extent (``pl.tensor.dim(x, 0)`` of the
    per-rank tensor), never the active count, so a padded batch, ``T < TP`` and a
    rank whose slab starts past the active range keep the same row mapping on both
    sides of every collective.  This mirrors
    ``decode_sp_integration.sequence_parallel_bounds`` on the CPU side; keep the
    two in step.
    """
    first = pl.cast(pl.min(tp_rank * slab, num_tokens), pl.INT32)
    count = pl.cast(pl.max(0, pl.min(slab, num_tokens - first)), pl.INT32)
    return first, count


def owner_rows(actual, rank: int, active: int, *, tp_size: int | None = None) -> tuple[int, int, int]:
    """Return ``(first, count, width)`` of *rank* inside a per-rank slab tensor.

    The CPU mirror of :func:`slab_owner` for goldens and comparators: ``actual`` is
    a ``[ranks, slab, ...]`` tensor, so its second extent is the physical slab and
    its first one the number of ranks that participate (``tp_size`` when several
    DP groups share the tensor).
    """
    width = actual.shape[1]
    world = tp_size or actual.shape[0]
    first, count, _ = sequence_parallel_bounds(active, world, rank % world, capacity=width * world)
    return first, count, width


@pl.jit.inline
def mhc_pre_norm(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    attn_norm_weight: pl.Tensor,
    attention_input: pl.Tensor,
    normalized_attention: pl.Tensor,
    owned_rows: pl.Scalar[pl.INT32],
):
    """Collapse the mHC streams and normalize the rows this call owns.

    The block runs between ``mhc_pre`` and the Attention operator, and it is the
    single implementation behind both the spec-driven boundary
    (:func:`attention_pre`) and the C1A compositions, which keep their mixes in
    the caller.  ``owned_rows`` is the caller's row count, never a global one:
    the replicated wiring passes the active batch size, the sequence-parallel
    wiring passes the rank's local count from :func:`slab_owner`.  It only masks
    the slab suffix, which is zeroed here so a published slab never carries a
    stale value in its padding rows.
    """
    mhc_pre(x_hc, incoming_pre_mix, attention_input)
    rms_norm(attention_input, attn_norm_weight, normalized_attention)
    zero_bf16_padding(normalized_attention, pl.min(owned_rows, pl.tensor.dim(normalized_attention, 0)))


@pl.jit.inline(auto_scope=False)
def attention_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    hc_attn_fn: pl.Tensor,
    hc_attn_scale: pl.Tensor,
    hc_attn_base: pl.Tensor,
    attn_norm_weight: pl.Tensor,
    attention_input: pl.Tensor,
    normalized_attention: pl.Tensor,
    attention_pre_mix: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Build the stable mHC-pre and RMSNorm boundary around any Attention leaf."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    mhc_pre_norm(x_hc, incoming_pre_mix, attn_norm_weight, attention_input, normalized_attention, num_tokens)
    return post_mix, residual_mix


def make_boundary_specs(args, sharded=False):
    """Create replicated mHC inputs, visible boundaries, and runtime scalars.

    With ``sharded`` the per-token boundaries keep only the rank's own slab of
    the active range (a contiguous split of ``tokens // tp`` rows) and the
    gather scratch that receives the full-range Attention input is added.
    """
    # Keep one uniform physical slab per rank.  The last rank(s) may contain
    # padding when the active token count is not divisible by TP; the runtime
    # scalar ``num_tokens`` and the collectives' first/count calculation keep
    # those rows out of the gather and reduce-scatter.
    local = (args.tokens + args.tp - 1) // args.tp if sharded else args.tokens
    generator = torch.Generator().manual_seed(args.seed + 1000)
    specs = []
    shapes = {
        "x_hc": [local, C.HC_MULT, C.D],
        "incoming_pre_mix": [local, C.HC_MULT],
        "hc_attn_fn": [C.MIX_HC, C.HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [C.MIX_HC],
        "attn_norm_weight": [C.D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= C.HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        if sharded and name in ("x_hc", "incoming_pre_mix"):
            # Sequence-parallel ranks own different slices of the batch, so the token
            # inputs must differ per rank: with replicated values a gather or
            # ReduceScatter that mis-orders the slabs would still compare equal.
            stacked = torch.randn([args.tp, *shape], generator=generator)
            if name == "incoming_pre_mix":
                stacked = torch.softmax(stacked, dim=-1)
        else:
            stacked = value.unsqueeze(0).repeat(args.tp, *([1] * len(shape)))
        specs.append(
            TensorSpec(name, [args.tp, *shape], value.dtype, init_value=stacked, resident="stacked")
        )
    for name, shape, dtype in (
        ("attention_output", [local, C.D], torch.bfloat16),
        ("attention_hidden", [local, C.HC_MULT, C.D], torch.float32),
        ("attention_pre_mix", [local, C.HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name == "attention_output" else 0.0
        specs.append(TensorSpec(name, [args.tp, *shape], dtype, init_value=sentinel, resident="stacked"))
    if sharded:
        specs.append(
            TensorSpec(
                "gathered", [args.tp, args.tokens, C.D], torch.bfloat16,
                init_value=13.0, resident="stacked",
            )
        )
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(
        ScalarSpec(
            "attention_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            benchmark_step=args.epochs if args.bench else None,
        )
    )
    return specs


def assemble_specs(args, leaf_specs, spec_names, aliases=None, sharded=False):
    """Combine one leaf's exact specs with the stable half-layer boundary specs."""
    aliases = aliases or {}
    specs = []
    for spec in leaf_specs:
        name = aliases.get(spec.name, spec.name)
        if name in ("x", "output", *SCALAR_NAMES):
            continue
        specs.append(spec if name == spec.name else replace(spec, name=name))
    specs.extend(make_boundary_specs(args, sharded))
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in spec_names if name not in by_name]
    if missing:
        raise ValueError(f"missing Attention specs: {missing}")
    return [by_name[name] for name in spec_names]


def check_program_specs(world_size, specs, spec_names):
    """Validate host ordering and return tensor shapes/dtypes keyed by name."""
    if world_size != C.TP_SIZE:
        raise ValueError("Attention world size must match TP_SIZE")
    if tuple(spec.name for spec in specs) != tuple(spec_names):
        raise ValueError("Attention specs must match the host parameter names and order")
    dtypes = {
        torch.bfloat16: pl.BF16,
        torch.float32: pl.FP32,
        torch.float8_e4m3fn: pl.FP8E4M3FN,
        torch.float8_e8m0fnu: pl.FP8E8M0,
        torch.int32: pl.INT32,
        torch.int64: pl.INT64,
        torch.uint8: pl.UINT8,
    }
    tensors = [spec for spec in specs if isinstance(spec, TensorSpec)]
    return SimpleNamespace(
        **{spec.name: SimpleNamespace(shape=spec.shape, dtype=dtypes[spec.dtype]) for spec in tensors}
    )


def golden_attention_pre(tensors):
    """Build normalized Attention inputs and return the mHC post state."""
    world = tensors["x_hc"].shape[0]
    active = int(tensors["num_tokens"])
    normalized, post, residual = [], [], []
    for rank in range(world):
        pre, post_mix, residual_mix = golden_mhc_mixes(
            tensors["x_hc"][rank],
            tensors["hc_attn_fn"][rank],
            tensors["hc_attn_scale"][rank],
            tensors["hc_attn_base"][rank],
        )
        tensors["attention_pre_mix"][rank].copy_(pre)
        collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
        normalized_rank = torch.full_like(collapsed, 13.0)
        normalized_rank[:active].copy_(golden_rms_norm(collapsed[:active], tensors["attn_norm_weight"][rank]))
        normalized.append(normalized_rank)
        post.append(post_mix)
        residual.append(residual_mix)
    return torch.stack(normalized), post, residual


def golden_attention_pre_sharded(tensors):
    """Build the local mHC/norm state and the gathered full-range input.

    Every rank collapses and normalizes its own slab of the active token range;
    the stacked return value carries the gathered rows each rank's leaf golden
    consumes, and ``tensors["gathered"]`` is filled so the harness can check the
    collector itself.
    """
    world = tensors["x_hc"].shape[0]
    active = int(tensors["num_tokens"])
    # Ownership follows the physical slab, while ``active`` only masks its
    # suffix.  This preserves rank mapping when active tokens are padded.
    normalized, post, residual = [], [], []
    for rank in range(world):
        pre, post_mix, residual_mix = golden_mhc_mixes(
            tensors["x_hc"][rank],
            tensors["hc_attn_fn"][rank],
            tensors["hc_attn_scale"][rank],
            tensors["hc_attn_base"][rank],
        )
        tensors["attention_pre_mix"][rank].copy_(pre)
        collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
        _first, count, _width = owner_rows(tensors["x_hc"], rank, active, tp_size=C.TP_SIZE)
        normalized_rank = torch.full_like(collapsed, 13.0)
        if count:
            normalized_rank[:count].copy_(
                golden_rms_norm(collapsed[:count], tensors["attn_norm_weight"][rank])
            )
        normalized.append(normalized_rank)
        post.append(post_mix)
        residual.append(residual_mix)
    per_rank_full = [None] * world
    for base in range(0, world, C.TP_SIZE):
        slabs = []
        for offset in range(C.TP_SIZE):
            rank = base + offset
            _first, count, _width = owner_rows(tensors["x_hc"], offset, active, tp_size=C.TP_SIZE)
            slabs.append(normalized[rank][:count])
        gathered = torch.cat(slabs, dim=0)[:active]
        for rank in range(base, base + C.TP_SIZE):
            tensors["gathered"][rank].zero_()
            tensors["gathered"][rank][: gathered.shape[0]].copy_(gathered)
            per_rank_full[rank] = gathered
    return torch.stack(per_rank_full), post, residual


def golden_attention_post(tensors, post, residual):
    """Populate the common post-Attention mHC boundary."""
    active = int(tensors["num_tokens"])
    sharded = "gathered" in tensors
    for rank in range(tensors["x_hc"].shape[0]):
        if sharded:
            _first, count, _width = owner_rows(tensors["attention_output"], rank, active, tp_size=C.TP_SIZE)
            tensors["attention_hidden"][rank].zero_()
            tensors["attention_output"][rank][count:].zero_()
            if count:
                tensors["attention_hidden"][rank][:count].copy_(
                    golden_mhc_post(
                        tensors["attention_output"][rank][:count],
                        tensors["x_hc"][rank][:count], post[rank][:count], residual[rank][:count]
                    )
                )
        else:
            tensors["attention_hidden"][rank].copy_(
                golden_mhc_post(
                    tensors["attention_output"][rank], tensors["x_hc"][rank], post[rank], residual[rank]
                )
            )


def compare_unchanged(name):
    """Compare non-owner storage byte-for-byte, including CPU FP8 scales."""

    def compare(actual, expected, **kwargs):
        return torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"{name}: non-owner state must stay exact"
        )

    return compare


def make_compare_attention_hidden(compare_output):
    """Build the common active/inactive mHC output comparison around a leaf comparator."""

    def compare_attention_hidden(actual, expected, *, inputs, **kwargs):
        active = int(inputs["num_tokens"])
        for rank in range(actual.shape[0]):
            passed, detail = compare_output(actual[rank, :active], expected[rank, :active], **kwargs)
            if not passed:
                return False, f"rank {rank}: {detail}"
            if active < actual.shape[1]:
                passed, detail = compare_output(actual[rank, active:], expected[rank, active:], **kwargs)
                if not passed:
                    return False, f"rank {rank} inactive mHC suffix: {detail}"
        return True, "active mHC precision and independent inactive suffix checks passed"

    return compare_attention_hidden


def active_rows_from_golden(expected):
    """Infer the batch's active token count from an ownership-padded golden.

    ``expected`` is one ``[ranks, slab, ...]`` tensor whose inactive slab rows
    the golden leaves at zero, so the last non-zero row of the last owning rank
    names the batch size.  It is the fallback for entries that do not hand the
    batch size to the comparator: the C1A compositions build their specs from
    tensors only, so the harness exposes no ``num_tokens`` scalar for them.
    Reading the count out of the reference is exact wherever the padding
    contract holds, and a batch's own rows are real data rather than all-zero
    placeholders.
    """
    width = expected.shape[1]
    rows = expected.reshape(expected.shape[0], width, -1)
    owned = (rows.abs().sum(dim=-1) > 0).sum(dim=-1)
    active = 0
    for rank, count in enumerate(owned.tolist()):
        if count:
            active = max(active, rank * width + min(count, width))
    return active


def compare_owner_rows(compare_output, *, name="per-rank", require_zero_padding=False, active=None):
    """Compare only the rows each rank owns inside its physical slab.

    The single implementation behind every sequence-parallel comparator: the
    ownership rule comes from :func:`owner_rows`, so a slab width, padding or
    rank-order change is fixed in one place.  The batch size comes from
    ``active`` when the caller knows it, else from the ``num_tokens`` scalar the
    harness exposes, else from the golden through :func:`active_rows_from_golden`
    -- never from the tensor's capacity, which overstates ownership whenever a
    slab ends partially filled.  ``require_zero_padding`` adds the
    materialization contract of a ReduceScatter output, whose inactive rows the
    kernel writes as zeros.
    """

    def compare(actual, expected, **kwargs):
        inputs = kwargs.get("inputs")
        num_tokens = active
        if num_tokens is None and inputs is not None:
            num_tokens = inputs.get("num_tokens")
        if num_tokens is None:
            num_tokens = active_rows_from_golden(expected)
        num_tokens = int(num_tokens)
        for rank in range(actual.shape[0]):
            _first, count, width = owner_rows(actual, rank, num_tokens, tp_size=C.TP_SIZE)
            if require_zero_padding and count < width and not bool((actual[rank][count:] == 0).all()):
                return False, f"rank {rank}: inactive {name} rows must be zero, not stale data"
            if count == 0:
                continue
            passed, detail = compare_output(actual[rank][:count], expected[rank][:count], **kwargs)
            if not passed:
                return False, f"rank {rank}: {detail}"
        return True, f"every rank's own {name} rows pass"

    return compare


def make_compare_attention_hidden_sharded(compare_output):
    """Compare every rank's local mHC rows under the leaf's own budget."""
    return compare_owner_rows(compare_output, name="mHC")


def make_compare_sharded_rows(compare_output):
    """Compare a per-rank ReduceScatter output and assert its inactive rows are zero."""
    return compare_owner_rows(compare_output, name="ReduceScatter", require_zero_padding=True)


def make_boundary_comparisons(compare_output, sharded=False):
    """Build comparisons shared by every mHC-Attention composition boundary."""
    return {
        "attention_hidden": (
            make_compare_attention_hidden_sharded(compare_output)
            if sharded
            else make_compare_attention_hidden(compare_output)
        ),
        "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
    }


def compare_attention_gather(actual, expected, **kwargs):
    """BF16 budget for the gathered Attention input.

    The rows were already rounded once in BF16 by the local collapse and
    RMSNorm, so a one-step BF16 disagreement is the expected noise floor; the
    comparator still catches a mis-mapped row or a wrong gather order.
    """
    a, e = actual.float(), expected.float()
    error = (a - e).norm() / e.norm().clamp_min(1e-12)
    print(
        f"[PRECISION] gathered rel_l2={error.item():.6g} "
        f"max_abs={(a - e).abs().max().item():.6g}"
    )
    budget = ratio_allclose(atol=2 ** -6, rtol=2 ** -6)
    return budget(actual, expected, **kwargs)


# Every decode Attention entry runs the replicated and the sequence-parallel
# wiring in one call. Their ABIs differ (the sharded side shards the token
# extent and adds ``gathered``), so any tree that is replayed or persisted
# belongs to exactly one wiring.
WIRING_REPLICATED = "replicated"
WIRING_SHARDED = "sharded"
WIRING_CHOICES = ("both", WIRING_REPLICATED, WIRING_SHARDED)


def add_wiring_argument(parser):
    """Add the wiring selector every decode Attention entry shares."""
    parser.add_argument(
        "--wiring", choices=WIRING_CHOICES, default="both",
        help=(
            "which wiring to run; a replay tree is namespaced per wiring "
            "(<dir>/replicated, <dir>/sharded) because the two ABIs differ"
        ),
    )


def selected_wirings(wiring: str) -> tuple[str, ...]:
    """Return the wirings one ``--wiring`` choice runs, in execution order."""
    if wiring not in WIRING_CHOICES:
        raise ValueError(f"unknown wiring {wiring!r}; expected one of {WIRING_CHOICES}")
    if wiring == "both":
        return (WIRING_REPLICATED, WIRING_SHARDED)
    return (wiring,)


def wiring_replay_dir(base: str | None, wiring: str) -> str | None:
    """Namespace a replay directory per wiring.

    The wiring name is the subdirectory, so ``--golden-data``/``--runtime-dir``
    can never hand one ABI the artifacts of the other: wiring ``W`` always reads
    ``<base>/W``, whether one wiring or both run in this invocation.
    """
    if base is None:
        return None
    if wiring not in (WIRING_REPLICATED, WIRING_SHARDED):
        raise ValueError(f"a replay directory needs a concrete wiring, got {wiring!r}")
    return str(Path(base) / wiring)


def make_parser(description, default_layer_id, default_tokens, cases, default_seed=17):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, choices=(1, 4), default=4)
    parser.add_argument("--layer-id", type=int, default=default_layer_id)
    parser.add_argument("--tokens", type=int, default=default_tokens)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--case", choices=cases, default=cases[0])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    add_wiring_argument(parser)
    return parser


def validate_args(parser, args, *, allow_inactive=False):
    if args.cpu_golden:
        parser.error("Block CPU goldens are provided by decode_layer.py")
    if args.stage == "block":
        parser.error("Block composition is provided by decode_layer.py")
    if not 1 <= args.tokens <= C.DECODE_MAX_TOKENS or not 1 <= args.epochs <= 1000:
        parser.error("tokens or epochs out of range")
    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    if not 1 <= args.active_tokens <= args.tokens or not 1 <= args.requests <= args.active_tokens:
        parser.error("require 1 <= requests <= active tokens <= tokens")
    if not allow_inactive and args.active_tokens != args.tokens:
        # The reference of this mode publishes every capacity row (caches, state,
        # output), so an inactive suffix cannot be compared yet. Refusing the shape
        # is better than entering a comparison whose expectation is wrong. C2A
        # reuse publishes only the active prefix and passes ``allow_inactive``.
        parser.error(
            "this Attention mode's reference publishes all capacity rows; "
            "--active-tokens needs a reference that publishes the active prefix"
        )
    args.dp, args.bench = 1, False
    if args.tp != C.TP_SIZE:
        parser.error("--tp must match the import-time tensor parallel configuration")
    devices = (
        list(range(args.tp))
        if args.device is None or args.compile_only
        else [int(device) for device in args.device.split(",")]
    )
    if len(devices) != args.tp or len(set(devices)) != args.tp or min(devices) < 0:
        parser.error("--device must name exactly TP distinct nonnegative device IDs")
    return devices


def validate_sp_tokens(parser, args):
    """Validate the physical slab without rejecting empty logical owners.

    Sequence parallel always allocates ``ceil(tokens / tp)`` rows per rank.
    A rank may therefore have zero valid rows; it still participates in the
    collective using its padding row(s).
    """
    if args.tokens < 1 or args.tp < 1:
        parser.error("sequence-parallel requires positive tokens and tp")
    # The mHC entries keep the token count in the host signature instead of a
    # runtime scalar, so they always run every capacity row.
    active_tokens = getattr(args, "active_tokens", args.tokens)
    if active_tokens > args.tokens:
        parser.error("active tokens cannot exceed physical token capacity")
    args.local_tokens = (args.tokens + args.tp - 1) // args.tp
    return args.local_tokens


def run_attention(args, program, specs, golden_fn, comparisons, kind_name, devices):
    result = run(
        fn=program,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn=comparisons,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    print(f"[HALF] layer={args.layer_id} kind={kind_name} work_dir={result.work_dir}")
    return result


__all__ = [
    "BOUNDARY_OUTPUT_NAMES",
    "BOUNDARY_PREFIX_NAMES",
    "SCALAR_NAMES",
    "active_rows_from_golden",
    "assemble_specs",
    "attention_pre",
    "check_program_specs",
    "compare_attention_gather",
    "compare_owner_rows",
    "compare_unchanged",
    "combine_validation",
    "golden_attention_post",
    "golden_attention_pre",
    "golden_attention_pre_sharded",
    "make_boundary_comparisons",
    "make_compare_attention_hidden",
    "make_compare_attention_hidden_sharded",
    "make_compare_sharded_rows",
    "make_parser",
    "mhc_pre_norm",
    "owner_rows",
    "run_attention",
    "selected_wirings",
    "slab_owner",
    "validate_args",
    "validate_sp_tokens",
    "wiring_replay_dir",
    "zero_bf16_padding",
]
