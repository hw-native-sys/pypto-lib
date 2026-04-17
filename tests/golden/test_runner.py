# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ``golden_data`` cache read-back in :func:`golden.run`.

These tests mock out ``pypto.ir.compile`` and ``pypto.runtime.execute_compiled``
so they run without a device.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from golden import TensorSpec, run
from golden.runner import _save_tensors


class _FakeCompiled:
    """Stand-in for CompiledProgram returned by ir.compile()."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir


@pytest.fixture
def three_kinds_specs():
    """TensorSpec trio covering pure input / pure output / inout."""
    return [
        TensorSpec("x", [4], torch.float32, init_value=torch.randn),           # pure input
        TensorSpec("y", [4], torch.float32, is_output=True),                   # pure output
        TensorSpec("state", [4], torch.float32, init_value=torch.zeros,        # inout
                   is_output=True),
    ]


@pytest.fixture
def populated_cache(tmp_path):
    """Populate {tmp_path}/in/ + {tmp_path}/out/ for the three_kinds_specs fixture."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state_in = torch.tensor([10.0, 20.0, 30.0, 40.0])
    y_golden = torch.tensor([2.0, 3.0, 4.0, 5.0])
    state_out = torch.tensor([11.0, 22.0, 33.0, 44.0])
    _save_tensors(tmp_path / "in", {"x": x, "state": state_in})
    _save_tensors(tmp_path / "out", {"y": y_golden, "state": state_out})
    return tmp_path


def _patch_compile_and_execute(compiled_dir: Path, write_outputs_positional=None):
    """Build context managers that stub out ``ir.compile`` and
    ``pypto.runtime.execute_compiled``.

    Args:
        compiled_dir: What `compiled.output_dir` should resolve to.
        write_outputs_positional: Optional list whose entries correspond 1:1 to
            the tensors passed to execute_compiled (matching the order of
            ``tensor_specs``).  Non-None entries are copied in-place into the
            corresponding tensor, simulating a correct kernel.
    """
    fake = _FakeCompiled(compiled_dir)

    def fake_execute(work_dir, tensors, **kwargs):
        if write_outputs_positional is None:
            return
        for tensor, value in zip(tensors, write_outputs_positional):
            if value is not None:
                tensor[:] = value

    return (
        patch("pypto.ir.compile", return_value=fake),
        patch("pypto.runtime.execute_compiled", side_effect=fake_execute),
    )


class TestGoldenDataCacheHit:
    """``golden_data`` points at a complete cache: skip generate + compute."""

    def test_hit_skips_generate_and_golden_fn(self, populated_cache, three_kinds_specs, tmp_path):
        """With cache hit: create_tensor and golden_fn must not run; validate passes."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        # Simulate a correct kernel: it writes the cached golden values back into
        # the y and state tensors so validate_golden passes.
        y_golden = torch.tensor([2.0, 3.0, 4.0, 5.0])
        state_out = torch.tensor([11.0, 22.0, 33.0, 44.0])
        write_outputs = [None, y_golden, state_out]  # [x, y, state]

        def golden_fn_should_not_run(tensors):
            pytest.fail("golden_fn must not run when golden_data is a complete cache")

        def _no_create_tensor(self):
            pytest.fail(f"TensorSpec.create_tensor must not run for {self.name}")

        compile_p, exec_p = _patch_compile_and_execute(compiled_dir, write_outputs)
        with compile_p, exec_p, patch.object(TensorSpec, "create_tensor", _no_create_tensor):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn_should_not_run,
                golden_data=str(populated_cache),
            )

        assert r.passed, f"unexpected failure: {r.error}"
        # Read-only: no data/ written under compiled.output_dir.
        assert not (compiled_dir / "data").exists()

    def test_hit_without_golden_fn_still_validates(
        self, populated_cache, three_kinds_specs, tmp_path,
    ):
        """golden_fn=None + golden_data set → validation still runs via loaded out/."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        # Same setup as the previous test but no golden_fn.
        y_golden = torch.tensor([2.0, 3.0, 4.0, 5.0])
        state_out = torch.tensor([11.0, 22.0, 33.0, 44.0])
        write_outputs = [None, y_golden, state_out]

        compile_p, exec_p = _patch_compile_and_execute(compiled_dir, write_outputs)
        with compile_p, exec_p:
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=None,
                golden_data=str(populated_cache),
            )

        assert r.passed, f"unexpected failure: {r.error}"

    def test_hit_with_mismatched_device_output_fails(
        self, populated_cache, three_kinds_specs, tmp_path,
    ):
        """If device writes values that differ from cached golden → validation fails."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        bad_y = torch.full((4,), 99.0)
        bad_state = torch.full((4,), -1.0)
        write_outputs = [None, bad_y, bad_state]

        compile_p, exec_p = _patch_compile_and_execute(compiled_dir, write_outputs)
        with compile_p, exec_p:
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=None,
                golden_data=str(populated_cache),
            )

        assert not r.passed
        assert "does not match golden" in (r.error or "")

    def test_hit_loads_inout_initial_value_from_in(
        self, populated_cache, three_kinds_specs, tmp_path,
    ):
        """Verify that the tensor handed to execute_compiled for the inout "state"
        is the value from in/state.pt, not a freshly created one."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        observed: dict[str, torch.Tensor] = {}

        def capture_execute(work_dir, tensors, **kwargs):
            # Positions: 0=x, 1=y, 2=state  (per three_kinds_specs order)
            observed["x"] = tensors[0].clone()
            observed["state"] = tensors[2].clone()
            # Make validate_golden pass so we reach the end.
            tensors[1][:] = torch.tensor([2.0, 3.0, 4.0, 5.0])    # y_golden
            tensors[2][:] = torch.tensor([11.0, 22.0, 33.0, 44.0])  # state_out

        fake = _FakeCompiled(compiled_dir)
        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=capture_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=None,
                golden_data=str(populated_cache),
            )

        assert r.passed
        torch.testing.assert_close(observed["x"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        # Inout's initial value was loaded from in/state.pt.
        torch.testing.assert_close(observed["state"], torch.tensor([10.0, 20.0, 30.0, 40.0]))


class TestGoldenDataCacheMiss:
    """``golden_data`` is set but incomplete: RunResult fails immediately."""

    def test_empty_dir_lists_all_missing(self, three_kinds_specs, tmp_path):
        empty = tmp_path / "empty_cache"
        empty.mkdir()
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()
        compile_p, exec_p = _patch_compile_and_execute(compiled_dir)
        with compile_p, exec_p:
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=lambda t: None,
                golden_data=str(empty),
            )

        assert not r.passed
        assert "golden_data is missing files" in (r.error or "")
        # All required files named in the error.
        for frag in ["x.pt", "y.pt", "state.pt"]:
            assert frag in r.error

    def test_partial_cache_still_fails(self, three_kinds_specs, tmp_path):
        """If out/ exists but in/ does not → still fail, and report the missing in/ paths."""
        partial = tmp_path / "partial"
        _save_tensors(partial / "out", {
            "y": torch.zeros(4),
            "state": torch.zeros(4),
        })
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()
        compile_p, exec_p = _patch_compile_and_execute(compiled_dir)
        with compile_p, exec_p:
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=None,
                golden_data=str(partial),
            )

        assert not r.passed
        assert "golden_data is missing files" in (r.error or "")
        assert str(partial / "in" / "x.pt") in r.error
        assert str(partial / "in" / "state.pt") in r.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
