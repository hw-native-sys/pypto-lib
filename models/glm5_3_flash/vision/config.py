# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vision-tower shape constants, mirrored from ``vision_config`` in the checkpoint."""

VISION_DEPTH = 24
VISION_HIDDEN = 1024
VISION_HEADS = 16
VISION_HEAD_DIM = VISION_HIDDEN // VISION_HEADS
VISION_INTER = 4096
VISION_PATCH = 14
VISION_TEMPORAL_PATCH = 2
VISION_IN_CHANNELS = 3
VISION_IMAGE_SIZE = 448
VISION_MERGE = 2
VISION_PROJ_INTER = 10240
VISION_RMS_NORM_EPS = 1e-5

# One flattened patch: in_channels x temporal_patch x patch x patch.
VISION_PATCH_IN = VISION_IN_CHANNELS * VISION_TEMPORAL_PATCH * VISION_PATCH * VISION_PATCH

IMAGE_TOKEN_ID = 154854
VIDEO_TOKEN_ID = 154855
IMAGE_START_TOKEN_ID = 154830
IMAGE_END_TOKEN_ID = 154831
VIDEO_START_TOKEN_ID = 154832
VIDEO_END_TOKEN_ID = 154833


__all__ = [
    "IMAGE_END_TOKEN_ID",
    "IMAGE_START_TOKEN_ID",
    "IMAGE_TOKEN_ID",
    "VIDEO_END_TOKEN_ID",
    "VIDEO_START_TOKEN_ID",
    "VIDEO_TOKEN_ID",
    "VISION_DEPTH",
    "VISION_HEADS",
    "VISION_HEAD_DIM",
    "VISION_HIDDEN",
    "VISION_IMAGE_SIZE",
    "VISION_IN_CHANNELS",
    "VISION_INTER",
    "VISION_MERGE",
    "VISION_PATCH",
    "VISION_PATCH_IN",
    "VISION_PROJ_INTER",
    "VISION_RMS_NORM_EPS",
    "VISION_TEMPORAL_PATCH",
]
