# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for all six GDN stages, on model data rather than noise.

Every stage is fed the float64 reference outputs of the stages before it -- the
`reference` module's chain, narrowed at each boundary to the dtype the kernels
actually exchange -- and its result is compared against the reference for that
stage under the criterion the reference harness uses
(`megagdn-pto/tests/utils.py: NumericalAccuracy`, relative Frobenius <= 1e-3).

Feeding each stage the REFERENCE chain rather than the previous kernel's device
output is deliberate: it isolates the stage. `--chain` runs the other way, each
stage on the previous KERNEL's output, which is what the deployed pipeline does;
a stage's score there is still against a reference recomputed from the inputs it
actually received, so the two modes differ in the input, not in the yardstick.
`--save-output` implies `--chain`, because the pair it writes -- the original
inputs and the final output -- is only an end-to-end result if the kernels
produced it end to end.

    python models/qwen3_8_27b/test_gdn_stages.py -p a2a3 -d 0
    python models/qwen3_8_27b/test_gdn_stages.py -p a2a3 -d 0 --chain
    python models/qwen3_8_27b/test_gdn_stages.py -p a2a3sim --seq-len 1024
    python models/qwen3_8_27b/test_gdn_stages.py -p a2a3 -d 0 --stages solve_tril,chunk_o

This is a device entry point, not a pytest case: it needs an NPU and a compile,
so it carries no `test_` functions and CI's `pytest tests/...` never collects it.
"""
import argparse
import importlib
import time

STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
          "chunk_h", "chunk_o")

from config import GDN_TILING, QWEN3_8_27B

D = QWEN3_8_27B.linear_value_head_dim
CHUNK = GDN_TILING.chunk

# Stages that read q or k, and so need to know how many QK heads there are.
# chunk_cumsum and solve_tril are per value head and read neither.
GQA_STAGES = ("scaled_dot_kkt", "wy_fast", "chunk_h", "chunk_o")


# What each stage writes. The harness stamps TensorSpec.direction from the
# compiled kernel's pl.Out parameters, so it is only readable after the run;
# the comparators have to be built before it.
OUTPUTS = {
    "chunk_cumsum": ("g_sum",),
    "scaled_dot_kkt": ("a_out",),
    "solve_tril": ("t_out",),
    "wy_fast": ("w_out", "u_out"),
    "chunk_h": ("state", "v_new"),
    "chunk_o": ("o_out",),
}


# Which of a stage's inputs the preceding kernels produce, for --chain:
#   spec name -> (producing stage, its output name)
CHAINED = {
    "scaled_dot_kkt": {"g_sum": ("chunk_cumsum", "g_sum")},
    "solve_tril": {"a_in": ("scaled_dot_kkt", "a_out")},
    "wy_fast": {"a_in": ("solve_tril", "t_out"),
                "g_sum": ("chunk_cumsum", "g_sum")},
    "chunk_h": {"w": ("wy_fast", "w_out"), "u": ("wy_fast", "u_out"),
                "g_sum": ("chunk_cumsum", "g_sum")},
    "chunk_o": {"state": ("chunk_h", "state"), "v": ("chunk_h", "v_new"),
                "g_sum": ("chunk_cumsum", "g_sum")},
}


def _comparators(stage: str, captured=None) -> dict:
    """megagdn's criterion on every output, optionally capturing the device result."""
    import reference

    def make(name):
        def compare(actual, expected, actual_outputs=None, **_kw):
            if captured is not None:
                for out_name, tensor in (actual_outputs or {}).items():
                    captured[out_name] = tensor.detach().cpu().clone()
            ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
            print(f"[stats] {name}: {detail}", flush=True)
            return ok, detail
        return compare

    return {name: make(name) for name in OUTPUTS[stage]}


def _chain_specs(stage: str, specs: list, produced: dict) -> list:
    """Replace every input a previous stage produced with that stage's device output."""
    import dataclasses

    out = []
    for spec in specs:
        src = CHAINED.get(stage, {}).get(spec.name)
        if src is None or src[1] not in produced:
            out.append(spec)
            continue
        tensor = produced[src[1]].to(spec.dtype)
        out.append(dataclasses.replace(spec, init_value=(lambda x=tensor: x)))
    return out


def check_stage(stage: str, t: int, h: int, hg: int, platform: str, device: int,
                produced: dict | None = None,
                chain: bool = False) -> tuple[bool, str]:
    """Compile, run and validate one stage. Returns (passed, detail).

    `produced` captures each stage's device outputs; `chain` is what decides
    whether they are also fed forward. --save-output needs the capture without
    the substitution, so the two cannot be the same switch.
    """
    from golden import run

    mod = importlib.import_module(stage)
    kernel_kw = dict(hg=hg) if stage in GQA_STAGES else {}
    fn = mod.build_kernel(t=t, h=h, d=D, chunk=CHUNK, **kernel_kw)
    specs = mod.build_tensor_specs(t=t, h=h, d=D, chunk=CHUNK, hg=hg)
    if chain and produced is not None:
        specs = _chain_specs(stage, specs, produced)

    result = run(
        fn=fn,
        specs=specs,
        golden_fn=getattr(mod, f"golden_gdn_{stage}"),
        config=dict(platform=platform, device_id=device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn=_comparators(stage, produced),
    )
    return bool(result.passed), (result.error or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int,
                        default=QWEN3_8_27B.linear_num_value_heads,
                        help="value heads H")
    parser.add_argument("--qk-heads", type=int,
                        default=QWEN3_8_27B.linear_num_key_heads,
                        help="QK heads Hg; pass the same value as --heads for no "
                             "grouping")
    parser.add_argument("--stages", type=str, default=",".join(STAGES))
    parser.add_argument("--save-output", type=str, default=None,
                        help="write the final chunk_o result and the inputs that "
                             "produced it here (torch .pt), for an end-to-end score "
                             "against an external reference. IMPLIES --chain: the "
                             "saved pair is only end to end if the kernels produced "
                             "it end to end")
    parser.add_argument("--chain", action="store_true",
                        help="feed each stage the previous KERNEL's device output "
                             "instead of the reference chain; each stage is still "
                             "scored against a reference recomputed from the inputs "
                             "it actually received")
    args = parser.parse_args()

    hg = args.qk_heads
    if args.heads % hg:
        parser.error(f"H={args.heads} must be divisible by Hg={hg}")
    stages = [s for s in args.stages.split(",") if s]
    # --save-output saves the ORIGINAL inputs beside the final output, so that pair
    # is only meaningful if every stage ran on the previous kernel's result. It
    # therefore implies --chain rather than quietly enabling it.
    chain = args.chain or bool(args.save_output)
    produced: dict | None = {} if chain else None
    results = []
    for stage in stages:
        print(f"\n=================== {stage} ===================", flush=True)
        started = time.time()
        ok, detail = check_stage(stage, args.seq_len, args.heads, hg,
                                 args.platform, args.device, produced,
                                 chain=chain)
        results.append((stage, ok, detail, time.time() - started))

    print("\n=================== summary ===================")
    print(f"T={args.seq_len} H={args.heads} Hg={hg} D={D} chunk={CHUNK} "
          f"platform={args.platform} inputs="
          f"{'previous kernel output' if chain else 'reference chain'}")
    for stage, ok, detail, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {stage:<16} ({secs:5.1f}s)"
              + (f"  {detail}" if not ok else ""))
    if args.save_output and produced is not None and "o_out" in produced:
        import torch

        import reference

        x = reference.make_inputs(args.seq_len, args.heads, D, hg)
        torch.save(dict(q=x["q"], k=x["k"], v=x["v"], g_in=x["g"],
                        beta=reference.to_hT(x["beta"]), o_dev=produced["o_out"]),
                   args.save_output)
        print(f"  saved the pipeline output and its inputs to {args.save_output}")
    failed = [s for s, ok, _, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} stages pass"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
