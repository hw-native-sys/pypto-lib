# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""List the parallel configurations an A5 device entry declares.

The a5-v41-flash job runs an entry once per configuration its own command line
accepts, so the matrix lives with the operator rather than in CI. This runs the
entry up to ``parse_args`` and reads the ``choices`` of its ``--tp`` and
``--dp`` options; every combination is one case, and the case needs TP x DP
cards. Each output line is ``<cards> <arguments>``, e.g. ``8 --tp 4 --dp 2``.
An entry that declares neither option prints a single ``1``.

Usage: python .github/scripts/a5_parallel_cases.py <entry.py>
"""

import argparse
import itertools
import runpy
import sys

PARALLEL_FLAGS = ("--tp", "--dp")


class _ParserReached(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


def _entry_parser(path):
    def stop(parser, *args, **kwargs):
        raise _ParserReached(parser)

    argparse.ArgumentParser.parse_args = stop
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    except _ParserReached as reached:
        return reached.parser
    raise SystemExit(f"{path}: the entry never parsed a command line")


def _values(parser, flag):
    action = parser._option_string_actions.get(flag)
    if action is None:
        return [None]
    return list(action.choices) if action.choices else [action.default]


def main():
    parser = _entry_parser(sys.argv[1])
    for values in itertools.product(*(_values(parser, flag) for flag in PARALLEL_FLAGS)):
        cards, arguments = 1, []
        for flag, value in zip(PARALLEL_FLAGS, values):
            if value is not None:
                cards *= int(value)
                arguments += [flag, str(value)]
        print(cards, *arguments)


if __name__ == "__main__":
    main()
