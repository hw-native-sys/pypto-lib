"""Compare named PyPTO and vLLM-Ascend A2/A3 intermediate dumps."""

from __future__ import annotations

import argparse

from .precision import compare_dump, load_intermediate_dump


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto", required=True, help="torch.save dictionary or golden data/out directory")
    parser.add_argument("--vllm", required=True, help="torch.save dictionary or vLLM dump directory")
    parser.add_argument("--case", help="case name when the vLLM dump contains multiple cases")
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    args = parser.parse_args()
    metrics = compare_dump(
        load_intermediate_dump(args.pypto), args.vllm, case=args.case,
        rtol=args.rtol, atol=args.atol,
    )
    failed = False
    for name, result in metrics.items():
        print(
            f"{name}: pass={result.within_tolerance} exact={result.exact} max_abs={result.max_abs:.6g} "
            f"mean_abs={result.mean_abs:.6g} relative_l2={result.relative_l2:.6g} cosine={result.cosine:.8f}"
        )
        failed |= not result.within_tolerance
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
