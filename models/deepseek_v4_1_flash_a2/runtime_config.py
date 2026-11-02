"""Runtime sizing controls for A2/A3 operator entry points."""

import sys


def _parse_parallel_size(name: str, default: int) -> int:
    """Read one integer parallel-size option without changing model config."""

    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


TP_SIZE = _parse_parallel_size("tp", 1)


__all__ = ["TP_SIZE"]
