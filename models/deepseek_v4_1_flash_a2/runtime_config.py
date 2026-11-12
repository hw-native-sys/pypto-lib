# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime sizing controls for A2/A3 operator and Decode entry points."""

from dataclasses import dataclass
import sys


def _parse_parallel_size(name: str, default: int) -> int:
    """Read one integer parallel-size option without changing model config."""

    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


TP_SIZE = _parse_parallel_size("tp", 1)


@dataclass(frozen=True)
class DecodeParallelConfig:
    """Decode topology contract; device allocation remains the caller's job."""

    dp: int = 8
    tp: int = 2
    ep: int = 16
    pp: int = 1
    pcp: int = 1
    dcp: int = 1
    enable_expert_parallel: bool = True
    use_sequence_parallel_moe: bool | None = None
    cache_block_size: int = 128
    enable_fused_mc2: bool = True
    enable_dsa_cp: bool = True
    enable_flashcomm1: bool = True
    enable_shared_expert_dp: bool = True

    def validate(self) -> "DecodeParallelConfig":
        values = {
            "dp": self.dp,
            "tp": self.tp,
            "ep": self.ep,
            "pp": self.pp,
            "pcp": self.pcp,
            "dcp": self.dcp,
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"Decode parallel sizes must be positive, got {values}")
        if self.pp != 1 or self.pcp != 1 or self.dcp != 1:
            raise ValueError("A2/A3 Decode currently requires PP=1, PCP=1, and DCP=1")
        if self.dp * self.tp != self.ep:
            raise ValueError(f"Decode requires EP=DP*TP, got DP={self.dp}, TP={self.tp}, EP={self.ep}")
        if self.cache_block_size != 128:
            raise ValueError("A2/A3 Decode requires block_size=128")
        if not self.enable_expert_parallel:
            raise ValueError("Decode acceptance requires enable_expert_parallel=true")
        return self

    def runtime_contract(self) -> dict[str, object]:
        """Return constructed-runtime fields without guessing unresolved values."""

        return {
            "dp": self.dp,
            "tp": self.tp,
            "ep": self.ep,
            "pp": self.pp,
            "pcp": self.pcp,
            "dcp": self.dcp,
            "enable_expert_parallel": self.enable_expert_parallel,
            "use_sequence_parallel_moe": self.use_sequence_parallel_moe
            if self.use_sequence_parallel_moe is not None
            else "runtime_capture_required",
            "cache_block_size": self.cache_block_size,
            "enable_fused_mc2": self.enable_fused_mc2,
            "enable_dsa_cp": self.enable_dsa_cp,
            "enable_flashcomm1": self.enable_flashcomm1,
            "enable_shared_expert_dp": self.enable_shared_expert_dp,
            "communication_groups": "runtime_capture_required",
            "request_scheduler": "runtime_capture_required",
            "status": "not_run",
        }


DECODE_PARALLEL = DecodeParallelConfig(
    dp=_parse_parallel_size("dp", 8),
    tp=_parse_parallel_size("tp", 2),
    ep=_parse_parallel_size("ep", 16),
    pp=_parse_parallel_size("pp", 1),
    pcp=_parse_parallel_size("pcp", 1),
    dcp=_parse_parallel_size("dcp", 1),
).validate()

DECODE_RUNTIME_CONTRACT = DECODE_PARALLEL.runtime_contract()


__all__ = ["DECODE_PARALLEL", "DECODE_RUNTIME_CONTRACT", "DecodeParallelConfig", "TP_SIZE"]
