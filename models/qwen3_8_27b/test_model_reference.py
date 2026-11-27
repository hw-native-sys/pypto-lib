# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the complete text-prefill reference against Qwen3.5's own model.

The production model has 27.78B parameters, so constructing it is neither a
unit test nor a useful way to isolate reference mistakes. This test keeps the
architecture-faithful while reducing dimensions and layer count: one Gated DeltaNet layer,
one gated-attention layer, dense SwiGLU on both, embedding, residual paths,
final norm and LM head. It instantiates `Qwen3_5ForConditionalGeneration`, runs
its text-only path, and compares every decoder-layer boundary and the logits.

The reference is float64 mathematical truth. Transformers deliberately takes
several statistics, the delta rule and attention softmax through fp32 even when
the module parameters are float64, so the gate sits above that rounding floor.

Host only: no NPU and no compile. `transformers` is optional and a missing
installation skips the check, as in `test_block_reference.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.qwen3_8_27b import reference
from models.qwen3_8_27b.config import QWEN3_8_27B, Qwen38Config


TOL = 1e-6
MAX_THREADS = 8


def tiny_config() -> Qwen38Config:
    """Architecture-exact two-layer configuration small enough for a host check."""
    return Qwen38Config(
        name="Qwen3.8-reference-test",
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        attention_bias=False,
    )


def transformers_model(cfg: Qwen38Config, seed: int, dtype: torch.dtype):
    """Build the official conditional-generation class with a tiny unused vision tower."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text = Qwen3_5TextConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        layer_types=list(cfg.layer_types),
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        partial_rotary_factor=cfg.partial_rotary_factor,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            # The sections sum to rotary_dim / 2. They are immaterial for text,
            # whose temporal, height and width position streams are identical.
            "mrope_section": [1, 1, 0],
        },
        attention_bias=cfg.attention_bias,
        attention_dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=False,
        use_cache=False,
    )
    text._attn_implementation = "eager"
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=cfg.hidden_size,
        num_position_embeddings=16,
    )
    composite = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=cfg.vocab_size - 3,
        video_token_id=cfg.vocab_size - 2,
        vision_start_token_id=cfg.vocab_size - 5,
        vision_end_token_id=cfg.vocab_size - 4,
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    return Qwen3_5ForConditionalGeneration(composite).to(dtype).eval()


def _tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    return output.detach()[0]


def module_forward(model, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the official text path and capture the same boundaries as the reference."""
    language_model = model.model.language_model
    got = {}
    handles = []

    def keep(name: str) -> Callable:
        def hook(_module, _inputs, output):
            got[name] = _tensor(output)

        return hook

    handles.append(language_model.embed_tokens.register_forward_hook(keep("embedding")))
    for i, layer in enumerate(language_model.layers):
        handles.append(layer.input_layernorm.register_forward_hook(keep(f"layers.{i}.input_norm")))
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        handles.append(mixer.register_forward_hook(keep(f"layers.{i}.mixer")))
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(keep(f"layers.{i}.post_attention_norm"))
        )
        handles.append(layer.mlp.register_forward_hook(keep(f"layers.{i}.mlp")))
        handles.append(layer.register_forward_hook(keep(f"layers.{i}.out")))
    handles.append(language_model.norm.register_forward_hook(keep("hidden")))
    handles.append(model.lm_head.register_forward_hook(keep("logits")))

    try:
        with torch.no_grad():
            model(input_ids=input_ids[None], use_cache=False, logits_to_keep=0)
    finally:
        for handle in handles:
            handle.remove()
    return got


def compare(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Relative Frobenius distance and maximum absolute difference in float64."""
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    diff = actual - expected
    denom = expected.norm()
    relative = float(diff.norm() / denom) if float(denom) else float(diff.norm())
    return relative, float(diff.abs().max())


def check(seq_len: int, chunk: int, seed: int, tol: float) -> bool:
    assert (
        QWEN3_8_27B.layer_types
        == (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
        * 16
    )
    assert QWEN3_8_27B.rotary_dim == 64
    cfg = tiny_config()
    model = transformers_model(cfg, seed, torch.float64)
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    input_ids = reference.make_text_inputs(seq_len, cfg, seed + 1)

    started = time.time()
    ours = reference.text_prefill(
        input_ids,
        weights,
        cfg,
        chunk,
        capture_intermediates=True,
    )
    reference_seconds = time.time() - started
    started = time.time()
    theirs = module_forward(model, input_ids)
    module_seconds = time.time() - started

    points = (
        "embedding",
        "layers.0.input_norm",
        "layers.0.mixer",
        "layers.0.post_attention_norm",
        "layers.0.mlp",
        "layers.0.out",
        "layers.1.input_norm",
        "layers.1.mixer",
        "layers.1.post_attention_norm",
        "layers.1.mlp",
        "layers.1.out",
        "hidden",
        "logits",
    )
    print(
        f"[reference] T={seq_len}, chunk={chunk}: ours {reference_seconds:.2f}s, "
        f"Transformers {module_seconds:.2f}s"
    )
    print(f"  {'point':<34} {'rel frob':>12} {'max |diff|':>12}")
    passed = True
    for name in points:
        relative, max_diff = compare(ours[name], theirs[name])
        ok = relative <= tol
        passed &= ok
        print(f"  {name:<34} {relative:>12.3e} {max_diff:>12.3e}  {'ok' if ok else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted and ignored: this check is host-only, but CI invokes every runnable
    # file in a model directory with the common platform and device flags.
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted for CI's uniform invocation; unused")
    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=TOL)
    args = parser.parse_args()

    if importlib.util.find_spec("transformers") is None:
        print(
            "SKIP: transformers is not installed; this check compares against "
            "Qwen3_5ForConditionalGeneration and has nothing to run without it"
        )
        return 0

    torch.set_num_threads(min(torch.get_num_threads(), MAX_THREADS))
    torch.backends.mkldnn.enabled = False
    passed = check(args.seq_len, args.chunk, args.seed, args.tol)
    print(f"\n{'PASS' if passed else 'FAIL'}: complete text-prefill reference")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
