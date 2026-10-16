# Wo-A MXFP8 interface

Wo-A quantizes the grouped BF16 activation after inverse RoPE, then runs
W8A8 MXFP8 and emits BF16 latent values for Wo-B.

| Operand | Logical shape | Storage |
| --- | --- | --- |
| Activation | `[T, G, K]` | BF16, dynamically quantized in groups of 32 |
| Weight | `[G, K, N]` | FP8 E4M3FN |
| Weight scale | `[G, K/32, N]` | E8M0, packed per group as `MX_B_NN` |
| Latent output | `[T, G, N]` | BF16 |

TP1 uses `G=8`, `K=4096`, `N=1024`. Device activation/output interfaces
flatten the final two axes; each weight scale group uses a 2D dispatch
view. Callers pass `wo_a_scale` alongside `wo_a`.

The loader transposes checkpoint weights `[G*N, K]` into `[G, K, N]`,
preserving FP8 bytes. It expands scales `[G*N/32, K/32]` across each block
of 32 output channels, transposes and packs each group independently.
No full BF16 weight expansion is required.

Activation quantization uses native OCP `scale_alg=0`: the exponent is
`floor(log2(amax)) - 8`, clamped to the E8M0 range. Encoded scale zero
produces zero payloads. E4M3FN payloads saturate at +/-448 and round to
nearest, ties to even. FP32 accumulation narrows to BF16 before Wo-B;
Wo-B precision and TP output-group ownership are unchanged.

Q-A uses sequential, scale-corrected group-32 FP32 partial sums before its
BF16 boundary to limit rounding differences amplified by later FP8 stages.
