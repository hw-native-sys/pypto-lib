# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python-only helpers for DeepSeek-V4 prefill layer EP2 standalone tests."""


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def ranked_init(spec, n_ranks, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        return torch.stack(values, dim=0).contiguous()

    return init


def build_ranked_layer_specs(
    attention_specs,
    moe_specs,
    n_ranks,
    x_next_shape,
    torch,
    TensorSpec,
    attention_name_map=None,
    n_experts_global=None,
):
    attention_name_map = attention_name_map or {}
    tensor_specs = []
    scalar_specs = []
    scalar_names = set()
    tensor_names = set()
    active_tokens = None
    for spec in attention_specs:
        if not isinstance(spec, TensorSpec) and spec.name == "num_tokens":
            active_tokens = int(spec.value.item())
            break
    layer_id_value = 0
    for spec in moe_specs:
        if not isinstance(spec, TensorSpec) and spec.name == "layer_id":
            layer_id_value = int(spec.value.item())
            break

    for spec in attention_specs:
        if isinstance(spec, TensorSpec):
            if spec.name == "x_out":
                continue
            spec_name = attention_name_map.get(spec.name, spec.name)
            tensor_specs.append(
                TensorSpec(
                    spec_name,
                    [n_ranks, *spec.shape],
                    spec.dtype,
                    init_value=ranked_init(spec, n_ranks, torch),
                    is_output=spec.is_output,
                )
            )
            tensor_names.add(spec_name)
        else:
            scalar_specs.append(spec)
            scalar_names.add(spec.name)

    for spec in moe_specs:
        if isinstance(spec, TensorSpec):
            if spec.name in {"x_hc", "x_next"} or spec.name in tensor_names:
                continue
            if spec.name == "tid2eid" and n_experts_global is not None:
                def init_tid2eid(spec=spec):
                    _, vocab, topk = spec.shape
                    ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                    ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                    table = ((ids * topk + ks) % n_experts_global).to(dtype=spec.dtype)
                    return table.unsqueeze(0).expand(n_ranks, -1, -1).contiguous()

                tensor_specs.append(
                    TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid, is_output=spec.is_output)
                )
            elif spec.name == "input_ids":
                def init_input_ids(spec=spec):
                    _, tokens = spec.shape
                    active = tokens if active_tokens is None else min(active_tokens, tokens)
                    rows = []
                    for rank in range(n_ranks):
                        ids = torch.arange(tokens, dtype=spec.dtype)
                        row = torch.roll(ids, shifts=rank)
                        if layer_id_value >= 3 and active < tokens:
                            row[active:] = -1
                        rows.append(row)
                    return torch.stack(rows, dim=0).contiguous()

                tensor_specs.append(
                    TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_input_ids, is_output=spec.is_output)
                )
            else:
                tensor_specs.append(spec)
            tensor_names.add(spec.name)
        else:
            if spec.name in scalar_names:
                continue
            scalar_specs.append(spec)
            scalar_names.add(spec.name)

    tensor_specs.append(TensorSpec("x_attn", x_next_shape, torch.bfloat16, is_output=True))
    tensor_specs.append(TensorSpec("x_next", x_next_shape, torch.bfloat16, is_output=True))
    return tensor_specs + scalar_specs


def golden_prefill_layer_ep(
    tensors,
    attention_specs,
    attention_golden,
    moe_golden,
    n_ranks,
    torch,
    TensorSpec,
    attention_name_map=None,
):
    attention_name_map = attention_name_map or {}
    x_attn = torch.zeros_like(tensors["x_hc"])

    for rank in range(n_ranks):
        attn_tensors = {}
        for spec in attention_specs:
            if isinstance(spec, TensorSpec):
                if spec.name == "x_out":
                    attn_tensors[spec.name] = x_attn[rank]
                else:
                    source_name = attention_name_map.get(spec.name, spec.name)
                    attn_tensors[spec.name] = tensors[source_name][rank]
            else:
                attn_tensors[spec.name] = tensors[spec.name]
        attention_golden(attn_tensors)

    tensors["x_attn"][:] = x_attn
    moe_x = x_attn.clone()
    num_tokens = int(tensors.get("num_tokens", moe_x.shape[1]))
    if num_tokens < moe_x.shape[1]:
        moe_x[:, num_tokens:] = tensors["x_hc"][:, num_tokens:]
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = moe_x
    moe_golden(moe_tensors)


def active_ranked_x_next_compare(num_tokens):
    from golden import ratio_reldiff

    base_compare = ratio_reldiff(diff_thd=0.1, pct_thd=0.05)

    def compare(actual, expected, **kwargs):
        return base_compare(actual[:, :num_tokens], expected[:, :num_tokens], **kwargs)

    compare.__name__ = f"active_ranked_x_next_compare(num_tokens={num_tokens})"
    return compare
