# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG Host control values; never copy device tensors to obtain these values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CSAHostMetadata:
    """Maximum *uncompressed* KV length, including this step's query tokens.

    Keep the device ``kv_seq_lens`` input: per-request lengths, padding and
    position-dependent masks remain device data. This scalar only selects the
    Host Score/Top-K task topology. Native already produces it on the CPU.
    """

    max_seq_len: int

    def __post_init__(self):
        if type(self.max_seq_len) is not int:
            raise TypeError("CSA HBG max_seq_len must be a Host int, not a device Tensor")
        if not 0 <= self.max_seq_len <= 2**31 - 1:
            raise ValueError("CSA HBG max_seq_len must fit a nonnegative INT32")

    @classmethod
    def from_native(cls, decode_metadata):
        # max_seq_lens is computed from the Native producer's CPU lengths.
        # Do not fall back to seq_lens.max().item(): that adds a device sync.
        return cls(decode_metadata.max_seq_lens)

    def graph_key(self) -> int:
        """Topology bucket; scalar values are frozen by an enclosing NPUGraph.

        Tensor lengths may change inside one bucket. Crossing a bucket requires
        a different capture; updating only device metadata cannot change Host
        branches in an already captured graph.
        """
        from ..common.config import INDEXER_NATIVE_CUBE_MIN_ROWS
        from .decode_indexer import COMPRESS_RATIO, TOPK_CANDIDATES_PER_LEAF

        rows = self.max_seq_len // COMPRESS_RATIO
        if rows < INDEXER_NATIVE_CUBE_MIN_ROWS:
            return 0
        return 1 if rows <= TOPK_CANDIDATES_PER_LEAF else 2

    def validate_replay(self, current: "CSAHostMetadata") -> None:
        if self.graph_key() != current.graph_key():
            raise ValueError(
                f"CSA HBG Host length changed Score/Top-K topology "
                f"({self.max_seq_len} -> {current.max_seq_len}); select or capture "
                "the matching graph, or use enforce_eager=True. Updating device "
                "seq_lens alone does not update a captured Host scalar."
            )


def validate_graph_replay(captured: dict[str, CSAHostMetadata], metadata: dict) -> None:
    """Called before model graph replay, which bypasses the Python CSA service."""
    for prefix, host in captured.items():
        if metadata is None or prefix not in metadata or metadata[prefix].decode is None:
            raise ValueError(f"CSA HBG graph replay is missing Native decode metadata for {prefix}")
        host.validate_replay(CSAHostMetadata.from_native(metadata[prefix].decode))
