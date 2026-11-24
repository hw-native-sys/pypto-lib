# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare the three stages of a V4.1 O projection in the A5 invocation that produces the output."""

import argparse
from functools import partial
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "models/deepseek_v4_1_flash"), str(REPO)]

import numpy as np
import torch

import golden
from golden import runner
from hc import golden_hc_post
import o_proj as o
from quant import _quantize_mxfp8_activation, encode_e8m0, pack_mx_a_scale, unpack_fp8_nz


def tag_boundaries(program):
    (host,) = program.glob("orchestration/*.cpp")
    tagged = {}
    lines = []
    stage = None
    index = 0
    for line in host.read_text().splitlines(keepends=True):
        if "// Group " in line:
            stage = line.split("// Group ")[1].split(":")[0]
            index = 0
        lines.append(line)
        arg = re.match(r"^(\s*)(params_\w+)\.add_(?:input|output|inout)\((\w+)\);", line)
        if not arg or stage is None:
            continue
        selected = (stage.startswith("proj_o_a_quant") and index in (0, 1, 2, 4)) or (
            stage == "_o_proj_b_hc_post" and index in (0, 2)
        )
        if selected:
            indent, params, tensor = arg.groups()
            lines.append(f"{indent}{params}.dump({tensor});\n")
            tagged.setdefault(stage, []).append(index)
        index += 1
    if tagged.get("_o_proj_b_hc_post") != [0, 2] or [0, 1, 2, 4] not in tagged.values():
        raise ValueError(f"Unrecognized O projection host ABI {tagged}")
    host.write_text("".join(lines))


def capture_stats(program, expected):
    manifest_file = program / "dfx_outputs/args_dump/args_dump.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest["dropped_records"] or manifest["truncated_args"]:
        raise ValueError("Incomplete O projection capture")
    hashes = {name: hashlib.sha256(buf).digest() for name, buf in expected.items()}
    stats = {}
    with (
        (manifest_file.parent / manifest["bin_file"]).open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw,
    ):
        for rec in manifest["args"]:
            if rec["truncated"] or not rec["is_contiguous"]:
                raise ValueError("Incomplete or strided O projection argument")
            dtype = {"FP8E4M3FN": "payload", "FP8E8M0": "scale"}[rec["dtype"]]
            if rec["stage"] == "after_completion":
                stage = "oa_out"
            elif rec["bin_size"] == len(expected["oa_in_" + dtype]):
                stage = "oa_in"
            else:
                stage = "ob_in"
            key = stage + "_" + dtype
            start = rec["bin_offset"]
            data = memoryview(raw)[start : start + rec["bin_size"]]
            state = stats.setdefault(key, {"captures": 0, "bad": 0})
            state["captures"] += 1
            if hashlib.sha256(data).digest() != hashes[key]:
                state["bad"] += 1
                if "first_byte" not in state:
                    bad = np.flatnonzero(
                        np.frombuffer(data, dtype=np.uint8) != np.frombuffer(expected[key], dtype=np.uint8)
                    )
                    pos = int(bad[0])
                    state.update(
                        different_bytes=int(bad.size),
                        first_byte=pos,
                        actual=data[pos],
                        expected=expected[key][pos],
                        task=rec["task_id"],
                    )
            del data
    if set(stats) != set(expected):
        raise ValueError(f"Missing captures {set(expected) - set(stats)}")
    return stats


def fixture(area, active):
    specs = o.build_o_proj_hc_post_specs(active, pad_to_tile=True)
    values = {
        s.name: (s.value if s.name == "num_tokens" else s.create_tensor())
        for s in specs
        if s.name != "output"
    }
    rows = values["attn_out"].shape[0]
    projected = torch.zeros(rows, o.D, dtype=torch.bfloat16)
    if active:
        attn = values["attn_out"][:active].reshape(active, o.LOCAL_H, o.HEAD_DIM)
        tail = o.rope_interleave(
            attn[..., -o.ROPE_DIM :], values["rope_cos"][:active], values["rope_sin"][:active], inverse=True
        )
        grouped = torch.cat((attn[..., : -o.ROPE_DIM], tail), dim=-1).reshape(
            active, o.LOCAL_O_GROUPS, o.O_GROUP_IN
        )
        latent = o.golden_grouped_output(grouped, values["wo_a"], values["wo_a_scale"]).flatten(-2)
        projected[:active] = o.golden_gemm_bf16_via_mxfp8_to_bf16_sequential(
            latent, unpack_fp8_nz(values["wo_b"]), values["wo_b_scale"]
        )
    normal = area / f"normal-{active}"
    (normal / "in").mkdir(parents=True)
    (normal / "out").mkdir()
    for name, value in values.items():
        torch.save(value, normal / "in" / f"{name}.pt")

    def save_hc(directory):
        expected = golden_hc_post(projected, values["residual"], values["post_mix"], values["residual_mix"])
        torch.save(expected, directory / "out/output.pt")

    save_hc(normal)
    if active != 17:
        return

    # Each head of HC has independent coefficients. The fourth can expose the
    # BF16 projection from the same vector register that feeds the first three.
    witness = area / "witness-17"
    (witness / "in").mkdir(parents=True)
    (witness / "out").mkdir()
    for name, value in values.items():
        target = witness / "in" / f"{name}.pt"
        if name in ("post_mix", "residual_mix"):
            value[..., 3] = int(name == "post_mix")
            torch.save(value, target)
        else:
            target.symlink_to(normal / "in" / f"{name}.pt")
    save_hc(witness)

    padded = 80
    a, sa = _quantize_mxfp8_activation(grouped, scale_alg=0)
    q, sq = _quantize_mxfp8_activation(latent, scale_alg=0)
    incoming = torch.zeros(o.LOCAL_O_GROUPS, padded, o.O_GROUP_IN, dtype=torch.uint8)
    incoming[:, :active] = a.view(torch.uint8).flatten(-2).transpose(0, 1)
    a_scale = torch.zeros(o.LOCAL_O_GROUPS, padded, o.O_GROUP_IN // 32, dtype=torch.uint8)
    a_scale[:, :active] = encode_e8m0(sa).transpose(0, 1)
    outgoing = torch.zeros(padded, o.LOCAL_O_WIDTH, dtype=torch.uint8)
    outgoing[:active] = q.view(torch.uint8).flatten(-2)
    q_scale = torch.zeros(padded, o.LOCAL_O_WIDTH // 32, dtype=torch.uint8)
    q_scale[:active] = encode_e8m0(sq)
    torch.save(
        dict(
            oa_in_payload=incoming,
            oa_in_scale=torch.stack([pack_mx_a_scale(x) for x in a_scale]),
            oa_out_payload=outgoing,
            oa_out_scale=pack_mx_a_scale(q_scale),
            projected=projected,
        ),
        area / "reference.pt",
    )


def prepare(area):
    for n in (0, 1, 17):
        fixture(area, n)
    result = golden.run(
        fn=o.o_proj_hc_post_validation,
        specs=o.build_o_proj_hc_post_specs(0, pad_to_tile=True),
        golden_fn=o.golden_o_proj_hc_post_validation,
        config={"backend_type": "a5", "execution_mode": False},
        compile_only=True,
    )
    if not result.passed:
        raise RuntimeError(result.error)
    src = Path(result.work_dir).resolve()
    program = area / "program"

    def ignore(directory, names):
        return [
            n
            for n in names
            if n == "dfx_outputs" or (Path(directory).name == "orchestration" and n.endswith(".so"))
        ]

    shutil.copytree(src, program, ignore=ignore)
    tag_boundaries(program)


def observe(mode, values, expected, projected, program, wanted):
    got = values["output"]
    gold = expected["output"]
    normal = got[:, :3] if mode == "witness" else got
    ref = gold[:, : normal.shape[1]]
    errors = ~torch.isclose(normal[:17], ref[:17], atol=0.01, rtol=1 / 128)
    record = {
        "active_elements": errors.numel(),
        "active_outliers": int(errors.sum()),
        "max_abs": float((normal[:17] - ref[:17]).abs().max()),
        "inactive_different": int((normal[17:] != ref[17:]).sum()),
    }
    if mode != "control":
        record["boundary"] = capture_stats(program, wanted)
    if mode == "witness":
        raw = got[:, 3]
        rebuilt = golden_hc_post(raw, values["residual"], values["post_mix"], values["residual_mix"])
        record.update(
            projection_different=int((raw != projected).sum()),
            projection_max=float((raw - projected).abs().max()),
            non_bf16=int((raw != raw.bfloat16().float()).sum()),
            hc_fma_different=int((normal != rebuilt[:, :3]).sum()),
        )
    indices = errors.nonzero()[:10].tolist()
    indices.extend([[0, 0, 0], [11, 1, 3676]])
    record["samples"] = []
    for row, stream, col in indices:
        point = {
            "index": [row, stream, col],
            "output": float(got[row, stream, col]),
            "reference": float(gold[row, stream, col]),
        }
        if mode == "witness":
            point.update(projected=float(raw[row, col]), projected_reference=float(projected[row, col]))
        record["samples"].append(point)
    return record


def run(area, device):
    reference = torch.load(area / "reference.pt", weights_only=True)
    projected = reference.pop("projected").float()
    wanted = {k: v.contiguous().numpy().tobytes() for k, v in reference.items()}
    wanted.update(ob_in_payload=wanted["oa_out_payload"], ob_in_scale=wanted["oa_out_scale"])
    program = area / "program"
    validate = runner._validate
    log = []
    failed = False
    for attempt in range(2):
        for mode in ("control", "capture", "witness"):
            for n in (0, 1, 17):
                folder = area / ("witness-17" if mode == "witness" and n == 17 else f"normal-{n}")
                specs = o.build_o_proj_hc_post_specs(n, pad_to_tile=True)
                fn = o.o_proj_hc_post_validation

                def inspect(sp, values, outputs, *args, **kwargs):
                    if n == 17:
                        rec = observe(mode, values, outputs, projected, program, wanted)
                        rec.update(attempt=attempt + 1, mode=mode, device=device)
                        log.append(rec)
                        print("O_CHAIN_PROBE " + json.dumps(rec, sort_keys=True), flush=True)
                        (area / "report.json").write_text(json.dumps(log, indent=2) + "\n")
                    return validate(sp, values, outputs, *args, **kwargs)

                runner._validate = inspect
                try:
                    result = runner._run_pipeline(
                        specs,
                        compile_step=lambda cfg, _: runner._reload_from_dir(program, cfg),
                        compile_label="Reload O projection",
                        prologue=runner._jit_entry(fn, specs)[1],
                        golden_fn=o.golden_o_proj_hc_post_validation,
                        golden_data=str(folder),
                        config=dict(
                            backend_type="a5",
                            execution_mode=False,
                            device_id=device,
                            enable_dump_args=int(n == 17 and mode != "control"),
                        ),
                        compare_fn={
                            "output": partial(
                                o.compare_o_proj_hc_post,
                                compare=golden.ratio_allclose(atol=0.01, rtol=1 / 128),
                            )
                        },
                        rtol=1e-5,
                        atol=1e-5,
                        compile_only=False,
                        runtime_dir=None,
                        save_data=False,
                        a5_default_runtime=None,
                    )
                finally:
                    runner._validate = validate
                if not result.passed:
                    failed = True
                    print(result.error, flush=True)
    if len(log) != 6 or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.getenv("TASK_DEVICE", "0")))
    args = parser.parse_args()
    area = args.work.resolve()
    area.mkdir(parents=True, exist_ok=True)
    os.chdir(area)
    if args.stage == "prepare":
        prepare(area)
    else:
        run(area, args.device)
