# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""首版TND服务的显式支持范围，只使用已有Host描述符。

query长度上界是当前DSpark服务配置，不能用它反算请求归属。历史下界保留
当前完整SPMD的Cube Score能力边界；更短历史仍待独立数值和设备验收，
在服务层回退Native，不借较大的max_model_len伪装成已支持。
"""

MAX_QUERY_TOKENS = 6
MIN_SEQUENCE_LENGTH = 8192
MAX_SEQUENCE_LENGTH = 1048576


def supported_descriptor(*, tokens, requests, actual_tokens, max_query_tokens, max_seq_len, batch_capacity):
    """输入均来自Native现成Host字段；不读device Q/L值，不要求等长请求。"""
    values = (tokens, requests, actual_tokens, max_query_tokens, max_seq_len, batch_capacity)
    if any(type(value) is not int for value in values):
        return False
    return (
        1 <= requests <= batch_capacity
        and 1 <= actual_tokens <= tokens
        and 1 <= max_query_tokens <= MAX_QUERY_TOKENS
        and actual_tokens <= requests * max_query_tokens
        and MIN_SEQUENCE_LENGTH <= max_seq_len <= MAX_SEQUENCE_LENGTH
    )
