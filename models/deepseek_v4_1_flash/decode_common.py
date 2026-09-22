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

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import D, HC_MULT
from models.deepseek_v4_1_flash.golden import rms_norm as golden_rms_norm
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.decode_sp_integration import combine_validation
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
    mhc_pre(x_hc, incoming_pre_mix, attention_input)
    rms_norm(attention_input, attn_norm_weight, normalized_attention)
    zero_bf16_padding(normalized_attention, pl.min(num_tokens, pl.tensor.dim(normalized_attention, 0)))
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
    width = tensors["x_hc"].shape[1]
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
        first = min((rank % C.TP_SIZE) * width, active)
        count = max(0, min(width, active - first))
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
            first = min(offset * width, active)
            count = max(0, min(width, active - first))
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
    width = tensors["attention_output"].shape[1] if sharded else tensors["attention_output"].shape[0]
    for rank in range(tensors["x_hc"].shape[0]):
        if sharded:
            first = rank * width
            count = max(0, min(width, active - first))
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


def make_compare_attention_hidden_sharded(compare_output):
    """Compare every rank's local mHC rows under the leaf's own budget."""

    def compare_attention_hidden(actual, expected, **kwargs):
        inputs = kwargs.get("inputs")
        active = int(inputs["num_tokens"]) if inputs is not None and "num_tokens" in inputs else actual.shape[1] * actual.shape[0]
        width = actual.shape[1]
        for rank in range(actual.shape[0]):
            first = min(rank * width, active)
            count = max(0, min(width, active - first))
            if count == 0:
                continue
            passed, detail = compare_output(actual[rank][:count], expected[rank][:count], **kwargs)
            if not passed:
                return False, f"rank {rank}: {detail}"
        return True, "every rank's local mHC rows pass the leaf budget"

    return compare_attention_hidden


def make_compare_sharded_rows(compare_output):
    """Compare only valid rows of a per-rank ReduceScatter output.

    The slab is fully materialized: rows inside the active range carry the summed
    rows this rank owns, rows outside it must be zero on every rank.
    """

    def compare(actual, expected, **kwargs):
        inputs = kwargs.get("inputs")
        active = int(inputs["num_tokens"]) if inputs is not None and "num_tokens" in inputs else actual.shape[1] * actual.shape[0]
        width = actual.shape[1]
        for rank in range(actual.shape[0]):
            first = min(rank * width, active)
            count = max(0, min(width, active - first))
            if count < width and not bool((actual[rank][count:] == 0).all()):
                return False, f"rank {rank}: inactive ReduceScatter rows must be zero, not stale data"
            if count == 0:
                continue
            passed, detail = compare_output(actual[rank][:count], expected[rank][:count], **kwargs)
            if not passed:
                return False, f"rank {rank}: {detail}"
        return True, "valid sequence-parallel rows pass"

    return compare


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
    "assemble_specs",
    "attention_pre",
    "check_program_specs",
    "compare_attention_gather",
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
    "run_attention",
    "validate_args",
    "validate_sp_tokens",
    "zero_bf16_padding",
]
