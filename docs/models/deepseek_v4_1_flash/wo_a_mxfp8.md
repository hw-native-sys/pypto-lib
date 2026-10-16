# Wo-A MXFP8 interface

The grouped output projection in
[`o_proj.py`](../../../models/deepseek_v4_1_flash/o_proj.py) applies W8A8
MXFP8 after inverse RoPE and emits BF16 latent values for Wo-B.

| Operand | Logical shape | Storage |
| --- | --- | --- |
| Post-inverse-RoPE activation | `[T, G, K]` | BF16 |
| Wo-A weight | `[G, K, N]` | FP8 E4M3FN |
| Wo-A weight scale | `[G, K/32, N]` | E8M0, independently packed per group as `MX_B_NN` |
| Wo-A latent output | `[T, G, N]` | BF16 |

For the TP1 Flash configuration, `G=8`, `K=4096`, and `N=1024`.
The device activation and output interfaces flatten the final two axes.
Each group's scale is dispatched as a two-dimensional `MX_B_NN` view.
The following Wo-B projection and its TP reduction retain their existing
precision and output-group ownership.

## Checkpoint layout

The checkpoint stores weights as `[G*N, K]` and E8M0 scales as
`[G*N/32, K/32]`. The loader preserves FP8 payload bytes, reshapes and
transposes the weights to `[G, K, N]`, expands each output-block scale
across 32 output channels, and packs each group's scales independently.
The MXFP8 path does not expand the full weight to BF16.

[`utils.py`](../../../models/deepseek_v4_1_flash/utils.py) stores resident
Wo-A payloads and scales in the FP8 banks. Attention, layer, and forward
interfaces pass the explicit `wo_a_scale` operand alongside `wo_a`.

## Activation quantization and rounding

Wo-A uses group-32 activation quantization with the native OCP default
`scale_alg=0`. For a nonzero group maximum `amax`, the scale exponent is
`floor(log2(amax)) - 8`, clamped to the representable E8M0 range. Payloads
saturate to the E4M3FN range and use round-to-nearest, ties-to-even.
Groups with encoded scale zero produce zero payloads, including all-zero
input groups. FP32 matrix accumulation narrows once to the BF16 latent
boundary.

Q-A uses sequential, scale-corrected group-32 FP32 partial sums in decode
and prefill before its BF16 boundary. The subsequent quantization can
amplify changes in BF16 rounding, so this accumulation order is part of
the documented numerical behavior.
