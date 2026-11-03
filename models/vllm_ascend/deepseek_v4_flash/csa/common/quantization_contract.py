# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Quantization formats whose loaded tensors satisfy the TND CSA contract."""


def validate_quantization_contract(method, config, variant):
    if method == "ascend":
        return
    if method != "compressed-tensors" or variant not in ("spmd_tnd", "mpmd_tnd"):
        raise ValueError("PTO CSA requires ModelSlim or symmetric dynamic W8A8 compressed-tensors with TND")
    if not isinstance(config, dict) or config.get("quant_method") != method or config.get("format") != "int-quantized":
        raise ValueError("TND CSA requires the int-quantized compressed-tensors format")
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("TND CSA requires explicit quantization groups")
    for group in groups.values():
        if not isinstance(group, dict) or group.get("targets") != ["Linear"]:
            raise ValueError("TND CSA currently supports the audited Linear-target W8A8 configuration")
        for name, strategy, dynamic in (("weights", "channel", False), ("input_activations", "token", True)):
            args = group.get(name)
            if not isinstance(args, dict) or not (
                args.get("num_bits") == 8 and args.get("type") == "int"
                and args.get("symmetric") is True and args.get("dynamic") is dynamic
                and args.get("strategy") == strategy
                and args.get("group_size") is None and args.get("block_structure") is None
            ):
                raise ValueError(f"TND CSA requires symmetric dynamic W8A8: incompatible {name}")
