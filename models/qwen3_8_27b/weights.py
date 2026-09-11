# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One real Gated DeltaNet layer of Qwen3.8-27B, fetched without the checkpoint.

The nine `linear_attn` tensors of a layer sit next to each other in their
safetensors shard, so a single HTTP range request of 221 MiB brings a whole
layer down without the 52 GiB checkpoint. The safetensors format is a JSON
header followed by the raw bytes, and the header (45 KiB, its own range request)
gives every tensor's dtype, shape and byte span.

    python models/qwen3_8_27b/weights.py                 # layer 0
    python models/qwen3_8_27b/weights.py --layer 2 --out my_layer2.pt
    python models/qwen3_8_27b/weights.py --quantize      # layer 0's W8A8 form

The file is a `torch.save` dict keyed by the tensor names under `linear_attn.`
-- `in_proj_qkv.weight`, `A_log`, ... -- in the checkpoint's bf16, which is what
`reference.block` and the module's `load_state_dict` both take. Default path:
`build_output/qwen3_8_27b/linear_attn_layer<N>.pt`, next to the compile output
and gitignored with it. `--quantize` writes the int8 form beside it
(`..._int8.pt`): the projections the kernels run as W8A8 replaced by int8 plus a
per-output-channel `weight_scale`, everything else as it was (see
`reference.quantize_weights`; the scheme is `config.GDN_QUANT`). The
checkpoint ships no int8, so this is the quantisation the kernels are scored
against until a converted checkpoint supplies its own scales.

Random weights (`reference.make_block_weights`) do not reach the decay range a
trained layer does, so this is the weight set that exercises `exp(g)` where the
kernels' fp16 clamps act. Layers 0, 1, 2 and 4 are in shard 1; any other
`--layer` reads the index to find its shard.
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import torch

REPO = "Qwen/Qwen3.8-27B"
# The commit `main` resolved to when the config was pinned (Q0). Byte offsets
# inside a shard belong to one upload, so the fetch names it rather than `main`.
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PREFIX = "model.language_model.layers.{layer}.linear_attn."

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def default_path(layer: int, quantized: bool = False) -> Path:
    root = Path(__file__).resolve().parents[2]
    suffix = "_int8" if quantized else ""
    return root / "build_output" / "qwen3_8_27b" / f"linear_attn_layer{layer}{suffix}.pt"


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=600) as r:
        if byte_range is not None and r.status != 206:
            raise RuntimeError(f"range request not honoured ({r.status}) for {url}")
        return r.read()


def fetch_layer(layer: int, repo: str = REPO, revision: str = REVISION,
                log=print) -> dict[str, torch.Tensor]:
    """The layer's `linear_attn.*` tensors, by three range requests: index, header, data."""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    prefix = PREFIX.format(layer=layer)

    index = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    shards = {index[name] for name in index if name.startswith(prefix)}
    if len(shards) != 1:
        raise RuntimeError(f"layer {layer}: expected one shard, found {sorted(shards)}")
    shard = shards.pop()

    header_len = struct.unpack("<Q", _get(base + shard, (0, 7)))[0]
    header = json.loads(_get(base + shard, (8, 8 + header_len - 1)))
    header.pop("__metadata__", None)
    entries = {name[len(prefix):]: info for name, info in header.items()
               if name.startswith(prefix)}
    lo = min(info["data_offsets"][0] for info in entries.values())
    hi = max(info["data_offsets"][1] for info in entries.values())
    log(f"[weights] layer {layer}: {len(entries)} tensors, {(hi - lo) / 2**20:.1f} MiB "
        f"from {shard} at revision {revision[:8]}")

    data_start = 8 + header_len
    blob = _get(base + shard, (data_start + lo, data_start + hi - 1))
    out = {}
    for name, info in entries.items():
        start, end = info["data_offsets"]
        raw = bytearray(blob[start - lo : end - lo])
        out[name] = torch.frombuffer(raw, dtype=_DTYPES[info["dtype"]]).reshape(info["shape"]).clone()
    return out


def load_layer(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer", type=int, default=0,
                        help="which of the model's linear-attention layers (default 0)")
    parser.add_argument("--out", type=str, default=None,
                        help="where to write the .pt (default build_output/qwen3_8_27b/)")
    parser.add_argument("--revision", type=str, default=REVISION)
    parser.add_argument("--quantize", action="store_true",
                        help="write the layer's W8A8 form instead, from the bf16 file "
                             "(fetched first if it is missing)")
    args = parser.parse_args()

    if args.quantize:
        import reference

        source = default_path(args.layer)
        if not source.is_file():
            weights = fetch_layer(args.layer, revision=args.revision)
            source.parent.mkdir(parents=True, exist_ok=True)
            torch.save(weights, source)
        weights = reference.quantize_weights(load_layer(source))
        out = Path(args.out) if args.out else default_path(args.layer, quantized=True)
    else:
        weights = fetch_layer(args.layer, revision=args.revision)
        out = Path(args.out) if args.out else default_path(args.layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    for name, w in weights.items():
        print(f"  {name:<24} {str(tuple(w.shape)):<16} {w.dtype}")
    print(f"[weights] saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
