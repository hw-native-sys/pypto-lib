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

    python models/gdn/test_gdn_stages.py -p a2a3 -d 0
    python models/gdn/test_gdn_stages.py -p a2a3 -d 0 --chain
    python models/gdn/test_gdn_stages.py -p a2a3sim --seq-len 1024
    python models/gdn/test_gdn_stages.py -p a2a3 -d 0 --stages solve_tril,chunk_o

This is a device entry point, not a pytest case: it needs an NPU and a compile,
so it carries no `test_` functions and CI's `pytest tests/...` never collects it.
"""
import argparse
import importlib
import time

STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
          "chunk_h", "chunk_o")

D = 128
CHUNK = 128


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


def _comparators(specs, captured=None) -> dict:
    """megagdn's criterion on every output, optionally capturing the device result."""
    from models.gdn import reference

    def make(name):
        def compare(actual, expected, actual_outputs=None, **_kw):
            if captured is not None:
                for out_name, tensor in (actual_outputs or {}).items():
                    captured[out_name] = tensor.detach().cpu().clone()
            ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
            print(f"[stats] {name}: {detail}", flush=True)
            return ok, detail
        return compare

    return {spec.name: make(spec.name) for spec in specs
            if getattr(spec, "is_output", False)}


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


def check_stage(stage: str, t: int, h: int, platform: str, device: int,
                produced: dict | None = None) -> tuple[bool, str]:
    """Compile, run and validate one stage. Returns (passed, detail)."""
    from golden import run_jit

    mod = importlib.import_module(f"models.gdn.{stage}")
    fn = mod.build_kernel(t=t, h=h, d=D, chunk=CHUNK)
    specs = mod.build_tensor_specs(t=t, h=h, d=D, chunk=CHUNK)
    if produced is not None:
        specs = _chain_specs(stage, specs, produced)

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=getattr(mod, f"golden_gdn_{stage}"),
        runtime_cfg=dict(platform=platform, device_id=device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn=_comparators(specs, produced),
    )
    return bool(result.passed), (result.error or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--stages", type=str, default=",".join(STAGES))
    parser.add_argument("--save-output", type=str, default=None,
                        help="write the final chunk_o result and the inputs that "
                             "produced it here (torch .pt), for an end-to-end score "
                             "against an external reference")
    parser.add_argument("--chain", action="store_true",
                        help="feed each stage the previous KERNEL's device output "
                             "instead of the reference chain; each stage is still "
                             "scored against a reference recomputed from the inputs "
                             "it actually received")
    args = parser.parse_args()

    stages = [s for s in args.stages.split(",") if s]
    produced: dict | None = {} if (args.chain or args.save_output) else None
    results = []
    for stage in stages:
        print(f"\n=================== {stage} ===================", flush=True)
        started = time.time()
        ok, detail = check_stage(stage, args.seq_len, args.heads,
                                 args.platform, args.device, produced)
        results.append((stage, ok, detail, time.time() - started))

    print("\n=================== summary ===================")
    print(f"T={args.seq_len} H={args.heads} D={D} chunk={CHUNK} "
          f"platform={args.platform} inputs="
          f"{'previous kernel output' if args.chain else 'reference chain'}")
    for stage, ok, detail, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {stage:<16} ({secs:5.1f}s)"
              + (f"  {detail}" if not ok else ""))
    if args.save_output and produced is not None and "o_out" in produced:
        import torch

        from models.gdn import reference

        x = reference.make_inputs(args.seq_len, args.heads, D)
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
