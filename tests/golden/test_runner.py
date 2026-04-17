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
from golden import RunConfig, TensorSpec, run
from golden.runner import RunResult, _backend_for_platform, _save_tensors


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


class TestGoldenFnPath:
    """No ``golden_data`` — the classic path that generates inputs, calls
    ``golden_fn``, and persists ``data/in/`` + ``data/out/`` under the
    compiled output directory."""

    def test_golden_fn_called_and_matches(self, three_kinds_specs, tmp_path):
        """``golden_fn`` runs, writes expected outputs, and validation passes."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        # golden_fn is called with a {name: tensor} dict — mutate y/state in place.
        def golden_fn(tensors):
            tensors["y"][:] = tensors["x"] + 1
            tensors["state"][:] = tensors["state"] + 100

        # execute_compiled must write the same values to the actual tensors.
        def fake_execute(work_dir, tensors, **_kwargs):
            # tensors positional: [x, y, state]; state was zero-initialized by
            # spec (init_value=torch.zeros), x was random.
            tensors[1][:] = tensors[0] + 1
            tensors[2][:] = tensors[2] + 100

        fake = _FakeCompiled(compiled_dir)
        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn,
            )

        assert r.passed, f"unexpected failure: {r.error}"
        # Persistence: data/in/ and data/out/ written under compiled.output_dir.
        assert (compiled_dir / "data" / "in" / "x.pt").is_file()
        assert (compiled_dir / "data" / "in" / "state.pt").is_file()
        assert (compiled_dir / "data" / "out" / "y.pt").is_file()
        assert (compiled_dir / "data" / "out" / "state.pt").is_file()

    def test_golden_fn_sees_cloned_inputs_not_live_tensors(
        self, three_kinds_specs, tmp_path,
    ):
        """``golden_fn`` receives a *clone* of inputs, not the live tensors
        handed to ``execute_compiled`` — so device writes don't corrupt the
        golden computation."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        captured = {}

        def golden_fn(tensors):
            captured["x_ptr"] = tensors["x"].data_ptr()
            tensors["y"][:] = tensors["x"] + 1
            tensors["state"][:] = tensors["state"] + 100

        device_x_ptrs = {}

        def fake_execute(work_dir, tensors, **_kwargs):
            device_x_ptrs["x"] = tensors[0].data_ptr()
            tensors[1][:] = tensors[0] + 1
            tensors[2][:] = tensors[2] + 100

        fake = _FakeCompiled(compiled_dir)
        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn,
            )

        assert r.passed
        # The golden_fn copy must not share storage with the device tensor.
        assert captured["x_ptr"] != device_x_ptrs["x"]

    def test_golden_fn_mismatch_fails(self, three_kinds_specs, tmp_path):
        """Device output diverges from golden_fn output → FAIL."""
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        def golden_fn(tensors):
            tensors["y"][:] = tensors["x"] + 1
            tensors["state"][:] = tensors["state"] + 100

        def bad_execute(work_dir, tensors, **_kwargs):
            tensors[1][:] = tensors[0] - 99  # wrong
            tensors[2][:] = tensors[2] + 100

        fake = _FakeCompiled(compiled_dir)
        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=bad_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn,
            )

        assert not r.passed
        assert "does not match golden" in (r.error or "")


class TestNoValidation:
    """Neither ``golden_fn`` nor ``golden_data`` — validation is skipped."""

    def test_skip_validation_passes_even_on_nonsense_outputs(
        self, three_kinds_specs, tmp_path,
    ):
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()

        def fake_execute(work_dir, tensors, **_kwargs):
            tensors[1][:] = torch.full_like(tensors[1], 9999.0)

        fake = _FakeCompiled(compiled_dir)
        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=None,
                golden_data=None,
            )

        assert r.passed
        # Inputs are still persisted (classic path), outputs are NOT computed/saved.
        assert (compiled_dir / "data" / "in" / "x.pt").is_file()
        assert not (compiled_dir / "data" / "out").exists()


class TestCompileOnly:
    """``RunConfig.compile_only`` short-circuits after compile."""

    def test_compile_only_skips_runtime_and_validation(
        self, three_kinds_specs, tmp_path,
    ):
        compiled_dir = tmp_path / "build"
        compiled_dir.mkdir()
        fake = _FakeCompiled(compiled_dir)

        def exec_must_not_run(*_args, **_kwargs):
            pytest.fail("execute_compiled must not run when compile_only=True")

        def golden_fn_must_not_run(_tensors):
            pytest.fail("golden_fn must not run when compile_only=True")

        with patch("pypto.ir.compile", return_value=fake), \
             patch("pypto.runtime.execute_compiled", side_effect=exec_must_not_run):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                config=RunConfig(compile_only=True),
                golden_fn=golden_fn_must_not_run,
            )

        assert r.passed
        assert r.error is None
        # compile_only path must not persist anything under data/.
        assert not (compiled_dir / "data").exists()


class TestRuntimeDir:
    """``runtime_dir`` skips compile and executes against a pre-compiled dir."""

    def test_runtime_dir_skips_compile(self, three_kinds_specs, tmp_path):
        """When runtime_dir is set, ir.compile must not be called and
        execute_compiled gets the runtime_dir as work_dir."""
        prebuilt = tmp_path / "prebuilt"
        prebuilt.mkdir()

        def compile_must_not_run(*_args, **_kwargs):
            pytest.fail("ir.compile must not run when runtime_dir is provided")

        observed_work_dir: list[Path] = []

        def fake_execute(work_dir, tensors, **_kwargs):
            observed_work_dir.append(Path(work_dir))
            # Leave outputs zero; no golden_fn is provided so no validation runs.

        with patch("pypto.ir.compile", side_effect=compile_must_not_run), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                runtime_dir=str(prebuilt),
            )

        assert r.passed, f"unexpected failure: {r.error}"
        assert observed_work_dir == [prebuilt]

    def test_runtime_dir_writes_data_under_runtime_dir(
        self, three_kinds_specs, tmp_path,
    ):
        """With golden_fn, data/in and data/out are persisted under runtime_dir."""
        prebuilt = tmp_path / "prebuilt"
        prebuilt.mkdir()

        def golden_fn(tensors):
            tensors["y"][:] = tensors["x"] + 1
            tensors["state"][:] = tensors["state"] + 100

        def fake_execute(_work_dir, tensors, **_kwargs):
            tensors[1][:] = tensors[0] + 1
            tensors[2][:] = tensors[2] + 100

        with patch("pypto.ir.compile", side_effect=lambda *a, **kw: pytest.fail("compile must not run")), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn,
                runtime_dir=str(prebuilt),
            )

        assert r.passed, f"unexpected failure: {r.error}"
        assert (prebuilt / "data" / "in" / "x.pt").is_file()
        assert (prebuilt / "data" / "in" / "state.pt").is_file()
        assert (prebuilt / "data" / "out" / "y.pt").is_file()
        assert (prebuilt / "data" / "out" / "state.pt").is_file()

    def test_runtime_dir_missing_returns_fail(self, three_kinds_specs, tmp_path):
        missing = tmp_path / "does_not_exist"

        def compile_must_not_run(*_args, **_kwargs):
            pytest.fail("ir.compile must not run when runtime_dir is provided")

        def exec_must_not_run(*_args, **_kwargs):
            pytest.fail("execute_compiled must not run when runtime_dir is missing")

        with patch("pypto.ir.compile", side_effect=compile_must_not_run), \
             patch("pypto.runtime.execute_compiled", side_effect=exec_must_not_run):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                runtime_dir=str(missing),
            )

        assert not r.passed
        assert "runtime_dir does not exist" in (r.error or "")

    def test_runtime_dir_with_compile_only_returns_fail(
        self, three_kinds_specs, tmp_path,
    ):
        prebuilt = tmp_path / "prebuilt"
        prebuilt.mkdir()

        def compile_must_not_run(*_args, **_kwargs):
            pytest.fail("ir.compile must not run")

        def exec_must_not_run(*_args, **_kwargs):
            pytest.fail("execute_compiled must not run")

        with patch("pypto.ir.compile", side_effect=compile_must_not_run), \
             patch("pypto.runtime.execute_compiled", side_effect=exec_must_not_run):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                config=RunConfig(compile_only=True),
                runtime_dir=str(prebuilt),
            )

        assert not r.passed
        assert "incompatible" in (r.error or "")

    def test_runtime_dir_with_golden_data_uses_runtime_dir_and_reads_cache(
        self, populated_cache, three_kinds_specs, tmp_path,
    ):
        """runtime_dir and golden_data are independent: execute uses runtime_dir,
        but inputs/goldens come from golden_data's cache (read-only)."""
        prebuilt = tmp_path / "prebuilt"
        prebuilt.mkdir()

        observed_work_dir: list[Path] = []

        def fake_execute(work_dir, tensors, **_kwargs):
            observed_work_dir.append(Path(work_dir))
            # Write the cached golden values so validation passes.
            tensors[1][:] = torch.tensor([2.0, 3.0, 4.0, 5.0])
            tensors[2][:] = torch.tensor([11.0, 22.0, 33.0, 44.0])

        def golden_fn_should_not_run(_tensors):
            pytest.fail("golden_fn must not run when golden_data is a complete cache")

        with patch("pypto.ir.compile", side_effect=lambda *a, **kw: pytest.fail("compile must not run")), \
             patch("pypto.runtime.execute_compiled", side_effect=fake_execute):
            r = run(
                program=object(),
                tensor_specs=three_kinds_specs,
                golden_fn=golden_fn_should_not_run,
                golden_data=str(populated_cache),
                runtime_dir=str(prebuilt),
            )

        assert r.passed, f"unexpected failure: {r.error}"
        assert observed_work_dir == [prebuilt]
        # golden_data cache-hit path is read-only: nothing is written under runtime_dir.
        assert not (prebuilt / "data").exists()


class TestBackendForPlatform:
    """``_backend_for_platform`` maps platform strings to BackendType values."""

    @pytest.mark.parametrize(
        "platform, expected_name",
        [
            ("a2a3", "Ascend910B"),
            ("a2a3sim", "Ascend910B"),
            ("a5", "Ascend950"),
            ("a5sim", "Ascend950"),
        ],
    )
    def test_known_platforms(self, platform, expected_name):
        backend = _backend_for_platform(platform)
        assert backend.name == expected_name

    def test_unknown_platform_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown runtime platform"):
            _backend_for_platform("notaplatform")


class TestRunResultStr:
    """``RunResult.__str__`` formatting — quick regression pins."""

    def test_pass_with_time(self):
        assert str(RunResult(passed=True, execution_time=1.234)) == "PASS (1.23s)"

    def test_fail_with_error_and_time(self):
        s = str(RunResult(passed=False, error="boom", execution_time=0.5))
        assert s == "FAIL: boom (0.50s)"

    def test_fail_without_error(self):
        assert str(RunResult(passed=False)) == "FAIL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
