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
entry up to ``parse_args`` and reads the ``choices`` of its ``--tp``, ``--dp``
and ``--ep`` options; every combination the entry accepts is one case. Each
output line is ``<cards> <arguments>``, e.g. ``8 --tp 4 --dp 2``. An entry that
declares none of them prints a single ``1``.

Entries come in two shapes. Most size their world as TP x DP; ``--ep``, where
they declare it at all, does not reach the run, so only ``--tp`` and ``--dp``
are sent. An expert-parallel entry instead sizes its world with ``--ep`` and
leaves DP implicit at EP / TP: it declares ``--ep`` and no ``--dp``, needs EP
cards rather than TP, and has to be sent ``--ep`` too, or it falls back to the
module default and rejects the card set the job borrowed. TP must divide EP
there, so the combinations that do not are not cases.

Usage: python .github/scripts/a5_parallel_cases.py <entry.py>
"""

import argparse
import itertools
import runpy
import sys


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
    tp_values = _values(parser, "--tp")
    dp_values = _values(parser, "--dp")
    ep_values = _values(parser, "--ep")
    # Only an entry that leaves DP implicit reads its world size off --ep.
    if dp_values != [None]:
        ep_values = [None]

    for tp, dp, ep in itertools.product(tp_values, dp_values, ep_values):
        cards, arguments = 1, []
        for flag, value in (("--tp", tp), ("--dp", dp)):
            if value is not None:
                cards *= int(value)
                arguments += [flag, str(value)]
        if ep is not None:
            # ``cards`` is TP here: EP is the world and DP is EP / TP, so a TP
            # that does not divide EP is not a configuration the entry accepts.
            if int(ep) % cards:
                continue
            cards = int(ep)
            arguments += ["--ep", str(ep)]
        print(cards, *arguments)


if __name__ == "__main__":
    main()
