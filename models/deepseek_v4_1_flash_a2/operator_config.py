"""A2/A3 operator shapes, tiling defaults, and dtype boundaries."""

from dataclasses import dataclass

import pypto.language as pl


# Dynamic shape variables.
T_DYN = pl.dynamic("V41_T_DYN")
B_DYN = pl.dynamic("V41_B_DYN")
STATE_BLOCKS_DYN = pl.dynamic("V41_STATE_BLOCKS_DYN")
Q_START_DYN = pl.dynamic("V41_Q_START_DYN")


@dataclass(frozen=True)
class A2A3OperatorConfig:
    """Shape and accumulation rules shared by A2/A3 operators."""

    int8_scale_max: float = 127.0
    int8_amax_epsilon: float = 1.0e-30
    int8_k_alignment: int = 32
    gemm_m_tile: int = 16
    gemm_n_tile: int = 64
    gemm_k_tile: int = 256
    rmsnorm_m_tile: int = 8
    swa_block_size: int = 128
    index_head_dim: int = 128
    index_scale_dtype: str = "float16"


A2A3_DTYPE_CONTRACT = {
    "hidden": "bfloat16",
    "weights": "int8",
    "activations": "int8",
    "gemm_accumulator": "int32",
    "weight_dequant_scale": "float32",
    "activation_dequant_scale": "float32",
    "rmsnorm_weight": "bfloat16",
    "rope_tables": "float32",
    "swa_cache": "bfloat16",
    "compressed_kv_cache": "bfloat16",
    "index_cache": "int8",
    "index_cache_scale": "float16",
    "expert_weights": "int8",
    "expert_scales": "float32",
}


CONTRACT = A2A3OperatorConfig()


__all__ = [
    "A2A3OperatorConfig",
    "A2A3_DTYPE_CONTRACT",
    "B_DYN",
    "CONTRACT",
    "Q_START_DYN",
    "STATE_BLOCKS_DYN",
    "T_DYN",
]
