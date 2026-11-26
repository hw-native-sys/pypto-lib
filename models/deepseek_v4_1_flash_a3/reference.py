# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent CPU W8A8 forward, causal cache state and frozen logit acceptance."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
import hashlib
import json
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import FLASH as M
from .compressor import golden_c1a_compressor, golden_c2a_compressor
from .engram import HostEngram, golden_engram
from .gate import golden_gate_reference
from .gemm import golden_activation_int8
from .hc import golden_hc_collapse, golden_hc_post, golden_mhc_mixes
from .host import prepare_prefill_inputs
from .o_proj import golden_o_proj
from .indexer_common import golden_select_candidate_block_pool
from .qkv_proj_rope import golden_rope_tail
from .quant import dequantize_mxfp4_cache, quantize_mxfp4_cache, quantize_mxfp8_cache
from .rmsnorm import golden_rms_norm
from .weights import Checkpoint


def int8_product(source, weight):
    """Accumulate exact INT8 products through bounded, exact FP32 partial sums."""
    if source.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("integer reference requires two INT8 matrices")
    if source.shape[-1] * 128 * 128 >= 2**31:
        raise ValueError("integer reference exceeds the INT32 accumulator range")
    # Each partial is bounded by 1024 * 128 * 128 = 2**24.
    # Every product and every partial sum is therefore an exact FP32 integer.
    result = torch.zeros((source.shape[0], weight.shape[1]), dtype=torch.int32)
    left, right = source.float(), weight.float()
    for begin in range(0, source.shape[-1], 1024):
        partial = left[:, begin:begin + 1024] @ right[begin:begin + 1024]
        result.add_(partial.to(torch.int32))
    return result


def quantize_token(value):
    """Return independently rounded symmetric token codes and FP32 scales."""
    scale = (value.float().abs().amax(-1, keepdim=True) * (1.0 / 127.0)).clamp_min(1e-30)
    return golden_activation_int8(value), scale


def quantized_projection(source, factor, weight, scale):
    """Apply checkpoint per-output scales after an exact integer dot product."""
    product = int8_product(source, weight)
    return (product.float() * factor * scale.reshape(1, -1)).bfloat16()


def selected_attention(query, window, window_ids, sink, compressed=None, compressed_ids=None, *, key_tile=64):
    """Evaluate bounded online BF16 PV tiles with a causal mask and one sink."""
    result = torch.empty_like(query)
    sources = [(window, window_ids)]
    if compressed is not None:
        sources.append((compressed, compressed_ids))
    for first in range(0, query.shape[0], 64):
        last = min(first + 64, query.shape[0])
        q = query[first:last].float()
        maximum = torch.full(q.shape[:2], -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        if compressed is None:
            maximum = sink.float().expand_as(maximum).clone()
            denominator.fill_(1.0)
        for cache, indices in sources:
            flat = cache.reshape(-1, M.head_dim)
            for begin in range(0, indices.shape[1], key_tile):
                ids = indices[first:last, begin:begin + key_tile].long()
                valid = ids >= 0
                kv = flat[ids.clamp_min(0)].bfloat16().float()
                kv.masked_fill_(~valid.unsqueeze(-1), 0)
                scores = torch.einsum("thd,tkd->thk", q, kv) * M.head_dim**-0.5
                scores.masked_fill_(~valid.unsqueeze(1), -1e30)
                next_maximum = torch.maximum(maximum, scores.amax(-1))
                correction = (maximum - next_maximum).exp()
                probability = (scores - next_maximum.unsqueeze(-1)).exp() * valid.unsqueeze(1)
                denominator = denominator * correction + probability.sum(-1)
                numerator = numerator * correction.unsqueeze(-1)
                numerator.add_(torch.einsum("thk,tkd->thd", probability.bfloat16().float(), kv))
                maximum = next_maximum
        if compressed is not None:
            final_maximum = torch.maximum(maximum, sink.float())
            correction = (maximum - final_maximum).exp()
            denominator = denominator * correction + (sink.float() - final_maximum).exp()
            numerator *= correction.unsqueeze(-1)
        result[first:last] = (numerator / denominator.unsqueeze(-1)).bfloat16()
    return result


class W8A8Reference:
    """Read original checkpoint matrices and preserve one request's CPU cache."""

    def __init__(self, checkpoint, tokenizer, *, save_dir=None, save_layers=()):
        self.checkpoint = checkpoint
        self.engram = HostEngram(checkpoint, tokenizer)
        self.cache = None
        self.rope = None
        self.committed = 0
        self.save_dir = save_dir
        self.save_layers = set(save_layers)
        self.trace = []
        self.overrides = {}

    def tensor(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return self.checkpoint.tensor(name)

    def linear(self, value, stem):
        payload, factor = quantize_token(value)
        weight = self.tensor(stem + ".weight").t().contiguous()
        offset = self.tensor(stem + ".weight_offset")
        if bool(offset.count_nonzero()):
            raise ValueError("reference requires symmetric checkpoint offsets")
        return quantized_projection(payload, factor, weight, self.tensor(stem + ".weight_scale"))

    def publish(self, name, slot, latent, slots, *, index=False):
        cache = self.cache[name + "_cache"][slot]
        scales = self.cache[name + "_scales"][slot]
        if name == "window":
            payload, factor = quantize_mxfp8_cache(latent)
        else:
            group, encoding = (32, "e8m0") if index else (16, "bf16")
            payload, factor = quantize_mxfp4_cache(latent, group, encoding, round_to_even=True)
        valid = slots >= 0
        cache.reshape(-1, cache.shape[-1])[slots[valid].long()] = payload.view(torch.uint8)[valid]
        scales.reshape(-1, scales.shape[-1])[slots[valid].long()] = factor[valid]

    def decode_cache(self, name, slot):
        payload = self.cache[name + "_cache"][slot]
        scale = self.cache[name + "_scales"][slot]
        if name == "window":
            return (payload.view(torch.float8_e4m3fn).float() * scale.float().repeat_interleave(32, -1)).bfloat16()
        group, encoding = (32, "e8m0") if name == "index" else (16, "bf16")
        return dequantize_mxfp4_cache(payload, scale, group, encoding).bfloat16()

    def index(self, x, latent, layer, cos, sin, lengths, source, *, pool=None):
        prefix = f"layers.{layer}.attn.indexer."
        query = self.linear(latent, prefix + "wq_b").reshape(-1, M.index_n_heads, M.index_head_dim)
        query = golden_rope_tail(query, cos, sin)
        code, factor = quantize_mxfp4_cache(query, 32, "e8m0", round_to_even=True)
        query = dequantize_mxfp4_cache(code, factor, 32, "e8m0").bfloat16()
        weights = F.linear(x.float(), self.tensor(prefix + "weights_proj.weight").float()).bfloat16()
        weights = (weights.float() * (M.index_n_heads * M.index_head_dim)**-0.5).bfloat16()
        width = int(lengths.max())
        keys = self.decode_cache("index", source).reshape(-1, M.index_head_dim)[:width].float().t()
        scores = torch.full((x.shape[0], width), -torch.inf)
        selected = torch.full((x.shape[0], M.index_topk), -1, dtype=torch.int32)
        positions = torch.arange(width)
        for first in range(0, x.shape[0], 32):
            last = min(first + 32, x.shape[0])
            dot = (query[first:last].float().flatten(0, 1) @ keys).reshape(last - first, M.index_n_heads, width)
            product = (dot.bfloat16().float().relu() * weights[first:last].float().unsqueeze(-1)).bfloat16()
            values = product.float().sum(1).bfloat16().float()
            valid = positions.unsqueeze(0) < lengths[first:last].unsqueeze(-1)
            if pool is not None:
                ids, counts = pool
                allowed = torch.zeros_like(valid)
                for local, token in enumerate(range(first, last)):
                    blocks = ids[token, :int(counts[token])].long()
                    logical = (blocks[:, None] * 8 + torch.arange(8)).flatten()
                    allowed[local, logical[(logical >= 0) & (logical < width)]] = True
                valid &= allowed
            values.masked_fill_(~valid, -torch.inf)
            scores[first:last] = values
            order = torch.argsort(values, dim=-1, descending=True, stable=True)[:, :M.index_topk]
            for local, token in enumerate(range(first, last)):
                count = min(M.index_topk, int(valid[local].sum()))
                selected[token, :count] = order[local, :count].sort().values.int()
        return scores, selected

    def attention(self, x, layer, values):
        plan = M.layer_config(layer)
        profile = int(plan.compression_ratio > 0)
        positions = values["position_ids"][0, :x.shape[0]].long()
        cos, sin = (table[profile, positions] for table in self.rope)
        prefix = f"layers.{layer}.attn."
        latent = golden_rms_norm(self.linear(x, prefix + "wq_a"), self.tensor(prefix + "q_norm.weight"))
        query = self.linear(latent, prefix + "wq_b").reshape(-1, M.num_attention_heads, M.head_dim)
        query = golden_rope_tail(query, cos, sin)
        kv = golden_rms_norm(self.linear(x, prefix + "wkv"), self.tensor(prefix + "kv_norm.weight"))
        kv = golden_rope_tail(kv, cos, sin)
        slots = values["window_slots"][0, :x.shape[0]]
        self.publish("window", layer, kv, slots)
        window = self.decode_cache("window", layer)
        window_ids = values["window_indices"][0, :x.shape[0]]
        compressed = None
        selected = None
        if plan.compression_ratio:
            source = M.kv_source_layer_ids.index(plan.kv_source_layer_id)
            lengths = ((positions + 1) // plan.compression_ratio).int()
            if plan.mode == "full":
                comp_prefix = prefix + "compressor."
                weight = self.tensor(comp_prefix + "wkv.weight").t()
                norm = self.tensor(comp_prefix + "norm.weight")
                if plan.compression_ratio == 1:
                    compact = golden_c1a_compressor(x, weight, norm)
                    publish_slots = values["c1a_compressed_slots"][:x.shape[0]]
                else:
                    compact, present = golden_c2a_compressor(
                        x, torch.tensor([0, x.shape[0]], dtype=torch.int32), positions.int(),
                        torch.zeros(x.shape[0], dtype=torch.int32), values["state_block_table"],
                        self.cache["state_cache"][source], weight,
                        self.tensor(comp_prefix + "wgate.weight").t(), norm,
                    )
                    publish_slots = values["c2a_compressed_slots"][:x.shape[0]].masked_fill(~present, -1)
                compressed_positions = positions // plan.compression_ratio * plan.compression_ratio
                comp_cos, comp_sin = (table[1, compressed_positions] for table in self.rope)
                rotated = golden_rope_tail(compact, comp_cos, comp_sin)
                self.publish("compressed", source, rotated, publish_slots)
                index_prefix = prefix + "indexer."
                key = F.linear(compact.float(), self.tensor(index_prefix + "wk.weight").float()).bfloat16()
                key = golden_rms_norm(key, self.tensor(index_prefix + "k_norm.weight"))
                key = golden_rope_tail(key, comp_cos, comp_sin)
                self.publish("index", source, key, publish_slots, index=True)
            if plan.mode != "reuse":
                pool = self.pool if plan.mode == "reindex" else None
                scores, self.selected = self.index(x, latent, layer, cos, sin, lengths, source, pool=pool)
                if layer == 20:
                    self.pool = golden_select_candidate_block_pool(scores, lengths)
            selected = self.selected
            compressed = self.decode_cache("compressed", source)
        sink = self.tensor(prefix + "attn_sink")
        key_tile = 128 if plan.compression_ratio == 1 and x.shape[0] > 1 else 64
        if compressed is None:
            key_tile = 128
        attended = selected_attention(query, window, window_ids, sink, compressed, selected, key_tile=key_tile)
        wo_a = self.tensor(prefix + "wo_a.weight").reshape(M.o_groups, M.o_lora_rank, -1).transpose(-1, -2)
        wo_b = self.tensor(prefix + "wo_b.weight").t()
        output = torch.zeros((x.shape[0], M.hidden_size))
        for lane in range(4):
            partial = golden_o_proj(attended[:, lane * 16:(lane + 1) * 16], cos, sin,
                                    wo_a[lane * 2:(lane + 1) * 2], wo_b[lane * 2048:(lane + 1) * 2048])
            output.add_(partial.float())
        return output.bfloat16()

    def expert(self, source, factor, prefix, route_weights=None):
        projections = []
        for name in ("w1", "w3"):
            weight = self.tensor(prefix + name + ".weight").t().contiguous()
            scale = self.tensor(prefix + name + ".weight_scale")
            projections.append(quantized_projection(source, factor, weight, scale).float())
        gate, up = projections
        hidden = F.silu(gate.clamp(max=M.swiglu_limit)) * up.clamp(-M.swiglu_limit, M.swiglu_limit)
        if route_weights is not None:
            hidden *= route_weights.reshape(-1, 1)
        return self.linear(hidden.bfloat16(), prefix + "w2")

    def moe(self, x, layer):
        prefix = f"layers.{layer}.ffn."
        route_weight, route_index = golden_gate_reference(
            x, self.tensor(prefix + "gate.weight"), self.tensor(prefix + "gate.bias"),
            accumulation_dtype=torch.float32,
        )
        payload, factor = quantize_token(x)
        shared = self.expert(payload, factor, prefix + "shared_experts.")
        routed = torch.empty((x.shape[0], M.num_experts_per_tok, M.hidden_size), dtype=torch.bfloat16)
        counts = torch.bincount(route_index.flatten().long(), minlength=M.n_routed_experts)
        for expert in counts.nonzero().flatten().tolist():
            rows, branches = (route_index == expert).nonzero(as_tuple=True)
            routed[rows, branches] = self.expert(payload[rows], factor[rows], prefix + f"experts.{expert}.",
                                                route_weight[rows, branches])
        output = shared.float()
        for branch in range(M.num_experts_per_tok):
            output.add_(routed[:, branch].float())
        return output.bfloat16(), counts

    @torch.inference_mode()
    def layer_forward(self, hidden, pre, layer, values):
        """Evaluate one attention and MoE layer from frozen residual inputs."""
        prefix = f"layers.{layer}."
        next_pre, post, residual = golden_mhc_mixes(
            hidden, self.tensor(prefix + "hc_attn_fn"), self.tensor(prefix + "hc_attn_scale"),
            self.tensor(prefix + "hc_attn_base"),
        )
        x = golden_rms_norm(golden_hc_collapse(hidden, pre).bfloat16(), self.tensor(prefix + "attn_norm.weight"))
        attention = self.attention(x, layer, values)
        hidden = golden_hc_post(attention, hidden, post, residual)
        pre, post, residual = golden_mhc_mixes(
            hidden, self.tensor(prefix + "hc_ffn_fn"), self.tensor(prefix + "hc_ffn_scale"),
            self.tensor(prefix + "hc_ffn_base"),
        )
        x = golden_rms_norm(golden_hc_collapse(hidden, next_pre).bfloat16(), self.tensor(prefix + "ffn_norm.weight"))
        ffn, counts = self.moe(x, layer)
        hidden = golden_hc_post(ffn, hidden, post, residual)
        return hidden, pre, counts

    @torch.inference_mode()
    def forward(self, ids, *, step):
        embedding = self.checkpoint.tensor_rows("embed.weight", torch.tensor(ids)).bfloat16()
        capacity = max(16, (len(ids) + 15) // 16 * 16)
        values = prepare_prefill_inputs(embedding, capacity, start_pos=self.committed, max_seq_len=8200,
                                        cache_state=self.cache, rope_tables=self.rope)
        if self.cache is None:
            self.cache = {name: value for name, value in values.items() if name.endswith(("_cache", "_scales"))}
            self.rope = values["freqs_cos"], values["freqs_sin"]
        lookup = self.engram.lookup(torch.tensor(ids), capacity, self.committed, request_id=0)[:, :len(ids)]
        hidden = embedding.float().unsqueeze(1).expand(-1, M.hc_mult, -1).contiguous()
        pre = torch.zeros((len(ids), M.hc_mult))
        pre[:, 0] = 1
        for layer in range(M.num_hidden_layers):
            started = time.monotonic()
            if layer in M.engram_layer_ids:
                prefix = f"layers.{layer}.engram."
                gate = self.tensor(prefix + "q_weight").float() * self.tensor(prefix + "k_weight").float()
                result = torch.empty_like(hidden)
                golden_engram(dict(source=hidden, lookup=lookup[M.engram_layer_ids.index(layer)],
                                   wkv_weight=self.tensor(prefix + "wkv.weight").t(), weight=gate,
                                   rotation=self.checkpoint.rotation_blocks(), output=result))
                hidden = result
            snapshot = None
            if step == 0 and layer in self.save_layers:
                snapshot = {"hidden": hidden.clone(), "pre_mix": pre.clone()}
            hidden, pre, counts = self.layer_forward(hidden, pre, layer, values)
            record = dict(step=step, layer=layer, route_count=int(counts.sum()), expert_counts=counts.tolist(),
                          seconds=time.monotonic() - started, residual_norm=float(hidden.norm()))
            self.trace.append(record)
            if snapshot is not None:
                snapshot.update(output=hidden, output_pre_mix=pre, counts=counts)
                torch.save(snapshot, self.save_dir / f"layer_{layer:02d}.pt")
            print("A3_REFERENCE_LAYER", json.dumps(record), flush=True)
        normalized = golden_rms_norm(golden_hc_collapse(hidden[-1:], pre[-1:]).bfloat16(), self.tensor("norm.weight"))
        logits = F.linear(normalized.float(), self.tensor("head.weight").float())[0]
        self.committed += len(ids)
        return logits


def compare_logits(actual, expected, policy):
    """Check every vocabulary value, frozen numerical limits and exact greedy IDs."""
    if actual.shape != expected.shape or actual.shape[-1] != M.vocab_size:
        raise ValueError("logit shape differs from the frozen complete vocabulary")
    a, b = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    relative_l2 = float((a - b).norm() / b.norm().clamp_min(1e-30))
    cosine = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0])
    top = torch.topk(b, 2)
    token = int(a.argmax())
    reference_token = int(top.indices[0])
    passed = finite and relative_l2 <= policy["relative_l2_max"] and cosine >= policy["cosine_min"]
    passed &= token == reference_token
    return dict(passed=passed, finite=finite, relative_l2=relative_l2, cosine=cosine,
                reference_margin=float(top.values[0] - top.values[1]), token_id=token,
                reference_token_id=reference_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-ids-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-layers", default="")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    request = json.loads(args.input_ids_json.read_text())
    ids = request["input_ids"]
    if len(ids) != 8192:
        parser.error("the reference replay must contain exactly 8192 input IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    policy = {"format": "deepseek-v41-w8a8-acceptance-v1", "relative_l2_max": 0.03, "cosine_min": 0.999,
              "greedy_tokens_required": 8, "finite_vocabulary_required": M.vocab_size,
              "checkpoint_identity": checkpoint.identity,
              "input_sha256": hashlib.sha256(args.input_ids_json.read_bytes()).hexdigest(),
              "reference": "CPU exact INT32 products, original checkpoint matrices, TP4 BF16 partials"}
    policy_path = args.output / "policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("frozen acceptance policy differs; use a new reference directory")
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    layers = [int(value) for value in args.save_layers.split(",") if value]
    reference = W8A8Reference(checkpoint, tokenizer, save_dir=args.output, save_layers=layers)
    generated, logits = [], []
    for step in range(8):
        output = reference.forward(ids if step == 0 else generated[-1:], step=step)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("nonfinite independent reference logits")
        generated.append(int(output.argmax()))
        logits.append(output)
        torch.save(dict(logits=torch.stack(logits), token_ids=generated, policy=policy), args.output / "logits.pt")
        (args.output / "trace.json").write_text(json.dumps(reference.trace, indent=2) + "\n")
        print("A3_REFERENCE_TOKEN", step + 1, generated[-1], flush=True)
    (args.output / "reference.json").write_text(json.dumps(dict(status="passed", token_ids=generated,
        text=tokenizer.decode(generated), committed=reference.committed, policy=policy), indent=2) + "\n")


if __name__ == "__main__":
    main()
