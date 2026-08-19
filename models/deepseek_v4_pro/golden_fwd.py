# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-network torch golden for the DeepSeek-V4 packed-prefill forward driver.

``golden_prefill_fwd(tensors)`` replays ``prefill_fwd.l3_prefill_fwd`` on host
by chaining the leaf goldens exactly the way the device program composes the
leaf kernels:

- ``input_pack.golden_pack_x_hc`` builds the layer-0 hidden state per rank.
- Per layer, the preset's attention kind runs per rank
  (``golden_prefill_attention_{swa,hca,csa}``) followed by the cross-rank
  ``moe.golden_moe`` whose ``x_next`` becomes the next layer's ``x_hc``.
- The trailing layer's MoE output is written straight into
  ``pre_hc_hidden_out``; ``hc_head`` + the final ``rms_norm`` produce
  ``hidden_out``, and a distributed-LM-head replay fills ``logits`` and
  ``sampled_ids``.

The *tensors* dict is the golden-harness scratch dict for
``prefill_fwd.build_tensor_specs()``: every tensor is ``[N_RANKS, ...]``
rank-stacked, per-FWD-layer weights and caches are packed along dim 1 by
FWD-layer id, and the CSA/HCA-compact stacks are packed by per-kind order
(ascending layer id of that kind). All cache slices handed to the leaf goldens
are torch views of the stacked tensors, so their in-place slot updates land in
the ``is_output`` cache tensors that validation reads back.
"""

# The prefill path runs PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe, which freezes recv shapes and derives RECV_MAX at import time (mirrors
# prefill_fwd.py; a no-op when prefill_fwd already imported moe).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS

# Import moe first: it applies the EP/active-preset override before the
# attention modules bake config-derived shapes (matches prefill_fwd's order).
from moe import D, HC_MULT, N_RANKS, T, golden_moe
from config import ACTIVE as MODEL_CONFIG
from prefill_attention_swa import golden_prefill_attention_swa
from prefill_attention_hca import HCA_STATE_BLOCK_SIZE, golden_prefill_attention_hca
from prefill_attention_csa import (
    CSA_STATE_BLOCK_SIZE,
    INNER_STATE_BLOCK_SIZE,
    golden_prefill_attention_csa,
)
from input_pack import golden_pack_x_hc
from hc_head import (
    LINEAR_K_PER_SPLIT as HC_HEAD_LINEAR_K_PER_SPLIT,
    LINEAR_K_CHUNK as HC_HEAD_LINEAR_K_CHUNK,
    LINEAR_OK as HC_HEAD_LINEAR_OK,
    RMS_K_CHUNK as HC_HEAD_RMS_K_CHUNK,
)
from lm_head import TP_SIZE as LM_HEAD_TP_SIZE
from rmsnorm import golden_rms_norm

# ---------------------------------------------------------------------------
# Layer schedule, mirrored from prefill_fwd: per-layer attention kind from the
# preset's compress ratios, plus each kind-compact stack's slot order
# (ascending layer id within the kind).
# ---------------------------------------------------------------------------
_KIND_BY_RATIO = {0: "swa", 4: "csa", 128: "hca"}
_ROPE_PROFILE_BY_KIND = {"swa": 0, "csa": 1, "hca": 1}
FWD_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_COMPRESS_RATIOS = MODEL_CONFIG.compress_ratios[:FWD_NUM_LAYERS]
LAYER_KINDS = tuple(_KIND_BY_RATIO[ratio] for ratio in FWD_COMPRESS_RATIOS)
CSA_NUM_LAYERS = LAYER_KINDS.count("csa")
HCA_NUM_LAYERS = LAYER_KINDS.count("hca")


def _kind_orders():
    orders = {}
    counters = {"swa": 0, "csa": 0, "hca": 0}
    for layer, kind in enumerate(LAYER_KINDS):
        orders[layer] = counters[kind]
        counters[kind] += 1
    return orders


KIND_ORDER = _kind_orders()


def _rope_profile_for_kind(stacked, kind):
    try:
        profile = _ROPE_PROFILE_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek V4 attention kind {kind!r}") from exc
    return stacked[profile]


def _layer_rows(stacked, count, index):
    """Packed-layer slice *index* along dim 1 of a rank-stacked tensor.

    Returns a torch view (never a copy) so leaf-golden in-place cache writes
    propagate into the stacked fwd tensor.
    """
    unit = stacked.shape[1] // count
    return stacked[:, index * unit:(index + 1) * unit]


# ---------------------------------------------------------------------------
# Per-layer view dicts: invert build_single_layer_tensor_specs' name mapping
# between the fwd namespace (csa_cmp_wkv, hca_cmp_ape, ...) and each leaf
# golden's own namespace (cmp_wkv, cmp_ape, ...).
# ---------------------------------------------------------------------------
def _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    """Keys shared by all three attention-kind goldens, for one rank/layer."""

    def fwd(name):
        return _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)[rank]

    kind = LAYER_KINDS[layer]
    return {
        "num_tokens": num_tokens,
        "x_hc": x_hc,
        "hc_attn_fn": fwd("hc_attn_fn"),
        "hc_attn_scale": fwd("hc_attn_scale"),
        "hc_attn_base": fwd("hc_attn_base"),
        "attn_norm_w": fwd("attn_norm_w"),
        "wq_a": fwd("wq_a"),
        "wq_b": fwd("wq_b"),
        "wq_b_scale": fwd("wq_b_scale"),
        "wkv": fwd("wkv"),
        "gamma_cq": fwd("gamma_cq"),
        "gamma_ckv": fwd("gamma_ckv"),
        "freqs_cos": _rope_profile_for_kind(tensors["freqs_cos"][rank], kind),
        "freqs_sin": _rope_profile_for_kind(tensors["freqs_sin"][rank], kind),
        "position_ids": tensors["position_ids"][rank],
        "kv_cache": fwd("kv_cache"),
        "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
        "attn_sink": fwd("attn_sink"),
        "wo_a": fwd("wo_a"),
        "wo_b": fwd("wo_b"),
        "wo_b_scale": fwd("wo_b_scale"),
        "x_out": x_out,
    }


def _swa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    # The SWA golden names the original-KV table plainly "block_table".
    views["block_table"] = tensors["ori_block_table"][rank]
    return views


def _hca_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def hca(name):
        return _layer_rows(tensors[name], HCA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": hca("hca_cmp_wkv"),
        "cmp_wgate": hca("hca_cmp_wgate"),
        "cmp_ape": hca("hca_cmp_ape"),
        "cmp_norm_w": hca("hca_cmp_norm_w"),
        "compress_state": hca("hca_compress_state"),
        "compress_state_block_table": tensors["hca_compress_state_block_table"][rank],
        "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"][rank],
        "state_slot_mapping": tensors["hca_state_slot_mapping"][rank],
    })
    return views


def _csa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def csa(name):
        return _layer_rows(tensors[name], CSA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": csa("csa_cmp_wkv"),
        "cmp_wgate": csa("csa_cmp_wgate"),
        "cmp_ape": csa("csa_cmp_ape"),
        "cmp_norm_w": csa("csa_cmp_norm_w"),
        "compress_state": csa("csa_compress_state"),
        "compress_state_block_table": tensors["csa_compress_state_block_table"][rank],
        "hadamard_idx": csa("csa_hadamard_idx"),
        "idx_wq_b": csa("csa_idx_wq_b"),
        "idx_wq_b_scale": csa("csa_idx_wq_b_scale"),
        "idx_weights_proj": csa("csa_weights_proj"),
        "inner_wkv": csa("csa_inner_wkv"),
        "inner_wgate": csa("csa_inner_wgate"),
        "inner_ape": csa("csa_inner_ape"),
        "inner_norm_w": csa("csa_inner_norm_w"),
        "inner_compress_state": csa("csa_inner_compress_state"),
        "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"][rank],
        "idx_kv_cache": csa("idx_kv_cache"),
        "idx_kv_scale": csa("idx_kv_scale"),
        "idx_block_table": tensors["idx_block_table"][rank],
        "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"][rank],
        "idx_slot_mapping": tensors["csa_idx_slot_mapping"][rank],
        "state_slot_mapping": tensors["csa_state_slot_mapping"][rank],
        "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"][rank],
    })
    return views


_ATTENTION_VIEWS = {
    "swa": _swa_attention_views,
    "hca": _hca_attention_views,
    "csa": _csa_attention_views,
}
_ATTENTION_GOLDEN = {
    "swa": golden_prefill_attention_swa,
    "hca": golden_prefill_attention_hca,
    "csa": golden_prefill_attention_csa,
}

_MOE_LAYER_STACKED = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
)


def _moe_views(tensors, layer, x_hc, x_next, num_tokens):
    """golden_moe consumes the full rank stack (dispatch/combine cross ranks)."""
    views = {
        name: _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)
        for name in _MOE_LAYER_STACKED
    }
    views.update({
        "x_hc": x_hc,
        "input_ids": tensors["input_ids"],
        "layer_id": layer,
        "num_tokens": num_tokens,
        "x_next": x_next,
    })
    return views


def _golden_hc_head_rows(tensors):
    """Row-count-agnostic port of ``hc_head.golden_hc_head``.

    The leaf golden bakes the decode row count (``hc_head.T`` = decode tokens)
    into its reshape, while the fwd feeds prefill ``T`` rows, so derive the row
    count from the input instead. The K-chunked accumulation order (RMS and
    linear mixes) and the HC_MULT==4 paired reduce are preserved bit-for-bit.
    """
    import torch

    x = tensors["x_hc"]
    rows = x.shape[0]
    hc_dim = MODEL_CONFIG.hc_dim
    x_flat_2d = x.reshape(rows, hc_dim).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum = torch.zeros(rows, 1, dtype=torch.float32)
    for k0 in range(0, hc_dim, HC_HEAD_RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + HC_HEAD_RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * (1.0 / hc_dim) + MODEL_CONFIG.rms_norm_eps)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(rows, 1, dtype=torch.float32)
        for linear_split in range(HC_HEAD_LINEAR_OK):
            split_col = torch.zeros(rows, 1, dtype=torch.float32)
            split_start = linear_split * HC_HEAD_LINEAR_K_PER_SPLIT
            split_end = split_start + HC_HEAD_LINEAR_K_PER_SPLIT
            for k0 in range(split_start, split_end, HC_HEAD_LINEAR_K_CHUNK):
                x_chunk = x_flat_2d[:, k0:k0 + HC_HEAD_LINEAR_K_CHUNK]
                w_chunk = hc_head_fn[h:h + 1, k0:k0 + HC_HEAD_LINEAR_K_CHUNK]
                split_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
            mix_col += split_col
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(rows, HC_MULT)

    pre = torch.sigmoid(
        mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()
    ) + MODEL_CONFIG.hc_eps
    x_view = x.float()
    if HC_MULT == 4:
        y = (
            x_view[:, 0, :] * pre[:, 0:1]
            + x_view[:, 1, :] * pre[:, 1:2]
        ) + (
            x_view[:, 2, :] * pre[:, 2:3]
            + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(rows, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h:h + 1]

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["y"][:] = y.to(torch.bfloat16)


def _golden_lm_head_all_ranks(tensors):
    """LM-head replay generalized to N_RANKS owners.

    Mirrors ``lm_head.golden_lm_head`` but iterates the fwd's EP world instead
    of the standalone fixture's DP_SIZE (which defaults to TP_SIZE and would
    fill only the first TP group's rows when EP > TP). Card ``r`` holds vocab
    shard ``r % TP_SIZE``, so concatenating the first TP_SIZE shards in index
    order reproduces the global vocabulary every owner assembles.
    """
    import torch

    hidden = tensors["hidden_out"].float()
    weight = tensors["lm_head_weight"].float()
    full_weight = torch.cat([weight[tp] for tp in range(LM_HEAD_TP_SIZE)], dim=0)
    logits = tensors["logits"]
    max_rows = logits.shape[1]
    for owner_rank in range(N_RANKS):
        selected = torch.zeros((max_rows, D), dtype=torch.float32)
        for row in range(max_rows):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        logits[owner_rank, :, :] = torch.matmul(selected, full_weight.t())
    tensors["sampled_ids"].zero_()
    tensors["sampled_ids"][:, :, 0] = torch.argmax(logits, dim=-1).to(torch.int32)


def golden_prefill_fwd(tensors):
    """Fill every ``is_output`` tensor of prefill_fwd's spec list in place."""
    import torch

    num_tokens = int(tensors["num_tokens"])

    # Layer-0 hidden state: embedding lookup replicated across the HC lanes.
    x_hc = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        golden_pack_x_hc({
            "input_ids": tensors["input_ids"][rank],
            "embed_weight": tensors["embed_weight"][rank],
            "x_hc": x_hc[rank],
        })

    for layer, kind in enumerate(LAYER_KINDS):
        attn_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        for rank in range(N_RANKS):
            views = _ATTENTION_VIEWS[kind](
                tensors, rank, layer, x_hc[rank], attn_out[rank], num_tokens,
            )
            _ATTENTION_GOLDEN[kind](views)
        # The trailing layer's MoE writes the fwd's pre-hc hidden output.
        if layer == FWD_NUM_LAYERS - 1:
            x_next = tensors["pre_hc_hidden_out"]
        else:
            x_next = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        golden_moe(_moe_views(tensors, layer, attn_out, x_next, num_tokens))
        x_hc = x_next

    # hc_head -> final rms_norm per rank, matching the tail of prefill_fwd.
    x_head = torch.zeros(N_RANKS, T, D, dtype=torch.bfloat16)
    for rank in range(N_RANKS):
        _golden_hc_head_rows({
            "x_hc": tensors["pre_hc_hidden_out"][rank],
            "hc_head_fn": tensors["hc_head_fn"][rank],
            "hc_head_scale": tensors["hc_head_scale"][rank],
            "hc_head_base": tensors["hc_head_base"][rank],
            "y": x_head[rank],
        })
        tensors["hidden_out"][rank].copy_(
            golden_rms_norm(x_head[rank], tensors["final_norm_w"][rank])
        )

    _golden_lm_head_all_ranks(tensors)


# ---------------------------------------------------------------------------
# --validate comparators. 43 chained layers accumulate error well past the
# leaf kernels' per-point bars, so the hidden/logits comparisons accept
# direction/magnitude agreement (bounded outlier ratio, cosine + relative-L2)
# instead of strict per-element closeness.
# ---------------------------------------------------------------------------
def _logits_cosine_compare(min_cosine=0.99, max_rel_l2=0.10):
    """Cosine + relative-L2 acceptance on the rows logit_row_indices selects."""

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        import torch

        del actual_outputs, expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    logits shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        row_indices = inputs["logit_row_indices"].cpu()
        a = actual.cpu().float()
        e = expected.cpu().float()
        failures = []
        for rank in range(a.shape[0]):
            for row in range(a.shape[1]):
                if int(row_indices[rank, row]) < 0:
                    continue
                a_row = a[rank, row]
                e_row = e[rank, row]
                if not bool(torch.isfinite(a_row).all()):
                    failures.append(f"    logits[{rank},{row}]: non-finite values")
                    continue
                denom = float(a_row.norm() * e_row.norm())
                cosine = float(a_row @ e_row) / denom if denom > 0.0 else 0.0
                rel_l2 = float((a_row - e_row).norm() / e_row.norm().clamp_min(1e-12))
                if cosine < min_cosine or rel_l2 > max_rel_l2:
                    failures.append(
                        f"    logits[{rank},{row}]: cosine={cosine:.6f} "
                        f"(min {min_cosine}) rel_l2={rel_l2:.6f} (max {max_rel_l2})"
                    )
        return (not failures), "\n".join(failures)

    cmp.__name__ = "logits_cosine_compare"
    return cmp


def build_validate_compare_fn(num_tokens):
    """Per-output comparators for prefill_fwd --validate.

    Hidden states resolve their active prefix from the run input, so replayed
    scalar data cannot be diluted by the spec initializer. Logits use
    cosine/relative-L2 acceptance on the selected rows, while sampled IDs must
    match both golden and the device-logit argmax. Layer-stacked pools compare
    only active mapped rows, independently per layer/rank, and require every
    other physical row to remain exactly equal to golden.
    """
    from golden import (
        input_prefix_ratio_allclose,
        sampled_ids_golden_compare,
        stacked_mapped_pool_ratio_allclose,
    )

    # Keep the public builder signature used by prefill_fwd. The comparator
    # intentionally reads the effective value from inputs["num_tokens"] at
    # validation time because golden-data replay overrides this initializer.
    del num_tokens
    hidden_cmp = input_prefix_ratio_allclose(
        "num_tokens",
        atol=1e-2, rtol=5e-2, max_error_ratio=0.02,
        valid_axis=1, exact_tail=True,
    )
    mapping_shape = (N_RANKS, T)
    all_layer_labels = tuple(range(FWD_NUM_LAYERS))
    csa_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "csa"
    )
    hca_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "hca"
    )

    def stacked_pool(
        mapping_names,
        *,
        layer_labels,
        block_size,
        pool_name,
        atol,
        rtol,
        max_error_ratio,
    ):
        return stacked_mapped_pool_ratio_allclose(
            tuple(mapping_names),
            mapping_shape=mapping_shape,
            block_size=block_size,
            active_rows_name="num_tokens",
            layer_labels=tuple(layer_labels),
            pool_name=pool_name,
            atol=atol,
            rtol=rtol,
            max_error_ratio=max_error_ratio,
        )

    return {
        "pre_hc_hidden_out": hidden_cmp,
        "hidden_out": hidden_cmp,
        "logits": _logits_cosine_compare(),
        "sampled_ids": sampled_ids_golden_compare(),
        "kv_cache": stacked_pool(
            ("ori_slot_mapping",) * FWD_NUM_LAYERS,
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="kv_cache",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "cmp_kv": stacked_pool(
            (
                None if kind == "swa" else f"{kind}_cmp_slot_mapping"
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="cmp_kv",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "hca_compress_state": stacked_pool(
            ("hca_state_slot_mapping",) * HCA_NUM_LAYERS,
            layer_labels=hca_layer_labels,
            block_size=HCA_STATE_BLOCK_SIZE,
            pool_name="hca_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_compress_state": stacked_pool(
            ("csa_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=CSA_STATE_BLOCK_SIZE,
            pool_name="csa_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_inner_compress_state": stacked_pool(
            ("csa_inner_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=INNER_STATE_BLOCK_SIZE,
            pool_name="csa_inner_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        # INT8 index cache: allow one quantization step on a bounded fraction.
        "idx_kv_cache": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_cache",
            atol=1,
            rtol=0,
            max_error_ratio=0.02,
        ),
        "idx_kv_scale": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_scale",
            atol=1e-3,
            rtol=1e-2,
            max_error_ratio=0.01,
        ),
    }
