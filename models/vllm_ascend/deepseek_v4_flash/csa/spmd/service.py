# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD policy and binding for the shared CSA service lifecycle."""

from ..common.weights import prepare_weights
from ..service import TNDServiceRuntime
from .binding import PreparedDynamicCSA
from .entry import decode_csa_spmd
from .policy import select_policy


class CSAServiceRuntime(TNDServiceRuntime):
    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_spmd)

    @staticmethod
    def _select_policy(**descriptors):
        return select_policy(**descriptors)

    @staticmethod
    def _prepare_binding(native, **options):
        return PreparedDynamicCSA(native, **options)
