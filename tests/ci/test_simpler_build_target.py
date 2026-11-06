# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that CI limits only the intended simpler wheel build."""

import os
from pathlib import Path
import subprocess
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SimplerBuildTargetTests(unittest.TestCase):
    def test_target_is_scoped_and_cache_key_is_distinct(self):
        action = (ROOT / ".github/actions/setup-ci-job/action.yml").read_text()
        start = action.index('        simpler_target="${SIMPLER_BUILD_TARGET:-}"')
        end = action.index('        # ----------------------------------------------------------------------', start)
        target_function = textwrap.dedent(action[start:end])
        start = action.index('        if [ -n "$simpler_target" ]; then\n          simpler_rt=')
        end = action.index('        fi', start) + len('        fi')
        cache_suffix = textwrap.dedent(action[start:end])
        script = (
            "set -euo pipefail\n" + target_function
            + "\nprobe() { python3 -c 'import os; print(os.environ.get(\"SKBUILD_BUILD_TARGETS\", \"UNSET\"))'; }\n"
            + "printf 'target='; with_simpler_target probe\nprintf 'outside='; probe\n"
            + "simpler_rt=sim-onboard-cann9\n" + cache_suffix + "\nprintf 'cache=%s\\n' \"$simpler_rt\"\n"
        )
        for target, outer, expected in [
            ("", None, ["target=UNSET", "outside=UNSET", "cache=sim-onboard-cann9"]),
            ("build_package_a2a3", None, ["target=build_package_a2a3", "outside=UNSET",
                                           "cache=sim-onboard-cann9-build_package_a2a3"]),
            ("build_package_a2a3", "outer", ["target=build_package_a2a3", "outside=outer",
                                              "cache=sim-onboard-cann9-build_package_a2a3"]),
        ]:
            with self.subTest(target=target, outer=outer):
                env = dict(os.environ, SIMPLER_BUILD_TARGET=target)
                env.pop("SKBUILD_BUILD_TARGETS", None)
                if outer is not None:
                    env["SKBUILD_BUILD_TARGETS"] = outer
                result = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env, check=True)
                self.assertEqual(result.stdout.splitlines(), expected)
        env = dict(os.environ, SIMPLER_BUILD_TARGET="unexpected")
        result = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported simpler build target", result.stdout)

    def test_pypto_build_remains_unscoped(self):
        action = (ROOT / ".github/actions/setup-ci-job/action.yml").read_text()
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn('rc=0; ensure_slot "$pypto_slot" "$PYPTO_SRC"', action)
        self.assertIn('rc=0; with_simpler_target ensure_slot "$simpler_slot" "$PYPTO_SRC/runtime"', action)
        self.assertEqual(action.count('with_simpler_target retry_build "$PYPTO_SRC/runtime"'), 2)
        self.assertIn("matrix.platform == 'a2a3' && 'build_package_a2a3' || ''", workflow)


if __name__ == "__main__":
    unittest.main()
