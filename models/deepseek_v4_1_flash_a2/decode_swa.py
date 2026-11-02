"""Compatibility exports for the shared A2/A3 SWA attention operator."""

from .swa_attention import make_swa_attention, swa_attention_reference


__all__ = ["make_swa_attention", "swa_attention_reference"]
