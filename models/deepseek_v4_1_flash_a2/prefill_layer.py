# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified A2/A3 Prefill layer and mHC boundary entry points."""

from .layer0 import layer0_reference, make_l0_attention, route_reference
from .layer0_single_card import make_ffn_post, make_ffn_pre, make_moe_input_quantize
from .layer2_c1c2 import make_layer2_c1c2
from .layer2_prefill import make_layer2_prefill_producer


__all__ = [
    "layer0_reference",
    "make_ffn_post",
    "make_ffn_pre",
    "make_l0_attention",
    "make_layer2_c1c2",
    "make_layer2_prefill_producer",
    "make_moe_input_quantize",
    "route_reference",
]
