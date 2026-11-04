# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only contracts for suite selection and prepared performance replay."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[2] / ".github/scripts/ci_perf.py"
_SPEC = importlib.util.spec_from_file_location("ci_perf", _PATH)
perf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf)


def _case(case_id="glm", suite="glm5.3-flash", platform="a2a3"):
    return {"case_id": case_id, "suite": suite, "platform": platform,
            "entrypoint": "models/glm5_3_flash/example.py", "device_count": 1,
            "arguments": ["--tp", "16", "--seed", "1024"]}


def test_suite_filter_keeps_legacy_default_and_platform(monkeypatch, tmp_path):
    default = _case("old"); default.pop("suite")
    manifest = {"platform": "a2a3", "sampling": {"rounds": 100, "warmup": 10},
                "cases": [default, _case(), _case("a5", "default", "a5")]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(perf, "MANIFEST", path)
    assert [x["case_id"] for x in perf.load_manifest("a2a3", "default")["cases"]] == ["old"]
    assert [x["case_id"] for x in perf.load_manifest("a2a3", "glm5.3-flash")["cases"]] == ["glm"]
    assert len(perf.load_manifest()["cases"]) == 3
    with pytest.raises(ValueError, match="no performance cases"):
        perf.load_manifest("a5", "glm5.3-flash")


def test_prepare_failure_never_allocates_device(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "load_manifest", lambda *_: {
        "sampling": {"rounds": 100, "warmup": 10}, "cases": [_case()],
    })
    monkeypatch.setattr(perf.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2))
    def allocate(*args, **kwargs):
        pytest.fail("CPU preparation failed but task-submit was still started")
    monkeypatch.setattr(perf.subprocess, "Popen", allocate)
    assert perf.run_ci(tmp_path / "results", "a2a3", "glm5.3-flash") == 1
    assert "FAIL" in (tmp_path / "results/published/report.md").read_text()


def test_glm_submission_replays_prepared_artifact(tmp_path):
    command = perf.submit_command(_case(), tmp_path, {"rounds": 100, "warmup": 10})
    assert command[:3] == ["task-submit", "--device", "auto"]
    assert "--prepared" in command[-1]
    assert str(tmp_path / "prepared.json") in command[-1]
    assert " prepare " not in command[-1]


@pytest.fixture
def prepared_environment(monkeypatch, tmp_path):
    case = _case()
    manifest = {"platform": "a2a3", "sampling": {"rounds": 2, "warmup": 1}, "cases": [case]}
    monkeypatch.setattr(perf, "ROOT", tmp_path)
    monkeypatch.setattr(perf, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(perf, "source_hashes", lambda *args: {"example.py": "fixed"})
    monkeypatch.setattr(sys, "argv", ["test"])
    build = tmp_path / "build"
    build.mkdir()
    (build / "kernel.bin").write_bytes(b"binary")
    events = []

    class TensorSpec:
        def __init__(self, name, direction):
            self.name, self.direction = name, direction
            self.shape, self.dtype, self.init_value = [2, 4], "float32", lambda: None

    class ScalarSpec:
        pass

    def comparator(*args, **kwargs):
        return True, ""

    def model_fn():
        pass

    def golden_fn(*args):
        pass

    stats = SimpleNamespace(
        rounds=2, warmup=1, fallback_flattened=False, all_zero_device=False,
        rounds_dispatches=[{0: [SimpleNamespace(effective_us=3)]}] * 2,
        per_round=lambda name: [5, 7] if name == "device" else [9, 11],
    )

    def run(**kwargs):
        if kwargs.get("compile_only"):
            events.append("compile")
            assert kwargs["config"]["execution_mode"] is False
            return SimpleNamespace(passed=True, error=None, work_dir=build)
        events.append("device")
        assert kwargs["config"]["device_id"] == 7
        assert kwargs["golden_fn"] is None
        assert kwargs["compare_fn"]["out"] is comparator
        assert all(spec.init_value is None for spec in kwargs["specs"])
        return SimpleNamespace(passed=True, error=None, bench=stats)

    def prepare_inputs(specs, tensors, scalars, data, target, save):
        events.append("inputs")
        directory = target / "data/in"; directory.mkdir(parents=True)
        (directory / "x.pt").write_bytes(b"input")
        return {}, {}

    def compute_golden(specs, tensors, scalars, inputs, target, data, fn, save):
        events.append("golden")
        assert fn is golden_fn
        directory = target / "data/out"; directory.mkdir(parents=True)
        (directory / "out.pt").write_bytes(b"reference")
        return {"out": SimpleNamespace(clone=lambda: b"reference")}

    def validate(specs, actual, expected, rtol, atol, callbacks, scalars):
        events.append("callback self-check")
        assert callbacks["out"] is comparator

    golden = ModuleType("golden")
    golden.run, golden.TensorSpec, golden.ScalarSpec = run, TensorSpec, ScalarSpec
    golden.runner = SimpleNamespace(_prepare_inputs=prepare_inputs, _compute_golden=compute_golden,
                                    _setup_runtime_dir=lambda *a, **k: build, _validate=validate)
    monkeypatch.setitem(sys.modules, "golden", golden)
    native = ModuleType("pypto.runtime.device_runner")
    class KernelCompiler:
        def _run(self, *args, **kwargs):
            return b"compiled"
    native.KernelCompiler = KernelCompiler
    native._compile_and_assemble = lambda *a, **k: events.append("assemble")
    monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", native)
    monkeypatch.setattr(sys.modules["pypto.runtime"], "device_runner", native, raising=False)
    pto_rebuild = ModuleType("pypto.runtime.debug.pto_rebuild")
    pto_rebuild._run_ptoas = lambda *a, **k: b"assembled"
    monkeypatch.setitem(sys.modules, "pypto.runtime.debug.pto_rebuild", pto_rebuild)
    monkeypatch.setattr(sys.modules["pypto.runtime.debug"], "pto_rebuild", pto_rebuild, raising=False)
    saved = {}
    serialization = ModuleType("cloudpickle")
    def dump(value, stream):
        saved["payload"] = value
        stream.write(b"private validation payload")
    serialization.dump = dump
    serialization.load = lambda stream: saved["payload"]
    monkeypatch.setitem(sys.modules, "cloudpickle", serialization)
    def entry(*args, **kwargs):
        golden.run(fn=model_fn, golden_fn=golden_fn,
                   specs=[TensorSpec("x", "in"), TensorSpec("out", "out")],
                   config={"backend_type": "a2a3", "device_id": 0}, compare_fn={"out": comparator})
    monkeypatch.setattr(perf.runpy, "run_path", entry)
    monkeypatch.setattr(perf.importlib, "import_module", lambda name: SimpleNamespace(model_fn=model_fn))
    return manifest, case, events, entry


def test_cpu_prepare_preserves_reference_and_replays_without_main(prepared_environment, tmp_path, monkeypatch):
    manifest, case, events, _ = prepared_environment
    path = tmp_path / "case/prepared.json"
    perf.prepare_case(SimpleNamespace(case="glm", output=path))
    assert events == ["compile", "assemble", "assemble", "inputs", "golden", "callback self-check"]
    def rerun_main(*args, **kwargs):
        pytest.fail("allocated replay reran model main or generated the CPU golden")
    monkeypatch.setattr(perf.runpy, "run_path", rerun_main)
    output = tmp_path / "case/result.json"
    perf.replay_prepared(manifest, case, [7], output, path)
    assert events[-1] == "device"
    result = json.loads(output.read_text())
    assert result["passed"]
    assert result["benchmark"]["device_wall_us"] == [5, 7]
    assert result["benchmark"]["samples"] == [{"0": [3]}, {"0": [3]}]


def test_prepared_replay_rejects_modified_golden(prepared_environment, tmp_path):
    manifest, case, events, _ = prepared_environment
    path = tmp_path / "case/prepared.json"
    perf.prepare_case(SimpleNamespace(case="glm", output=path))
    (path.parent / "frozen/data/out/out.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared artifact changed"):
        perf.replay_prepared(manifest, case, [7], tmp_path / "result.json", path)
    assert "device" not in events


def test_prepared_correctness_failure_preserves_result_and_fails(prepared_environment, tmp_path, monkeypatch):
    manifest, case, _, _ = prepared_environment
    path = tmp_path / "case/prepared.json"
    perf.prepare_case(SimpleNamespace(case="glm", output=path))
    monkeypatch.setattr(sys.modules["golden"], "run",
                        lambda **kwargs: SimpleNamespace(passed=False, error="output mismatch"))
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="correctness failed: output mismatch"):
        perf.replay_prepared(manifest, case, [7], output, path)
    saved = json.loads(output.read_text())
    assert saved["passed"] is False and saved["error"] == "output mismatch"
    assert "benchmark" not in saved


def test_prepare_rejects_multiple_golden_runs(prepared_environment, tmp_path, monkeypatch):
    _, _, events, entry = prepared_environment
    def two_calls(*args, **kwargs):
        entry(); entry()
    monkeypatch.setattr(perf.runpy, "run_path", two_calls)
    path = tmp_path / "case/prepared.json"
    with pytest.raises(ValueError, match="exactly once"):
        perf.prepare_case(SimpleNamespace(case="glm", output=path))
    assert not path.exists()
    assert "device" not in events


def test_cached_registration_is_allowed_but_actual_compilation_is_forbidden(prepared_environment):
    compiler = sys.modules["pypto.runtime.device_runner"].KernelCompiler()
    def register(cached_binary):
        return cached_binary if cached_binary is not None else compiler._run(["c++"], None, "Orchestration")
    with perf.cached_native_only():
        assert register(b"existing sidecar") == b"existing sidecar"
        with pytest.raises(RuntimeError, match="requires native compilation"):
            register(None)
        with pytest.raises(RuntimeError, match="requires native compilation"):
            sys.modules["pypto.runtime.debug.pto_rebuild"]._run_ptoas("ptoas", None, None)
    assert register(None) == b"compiled"


def test_fixed_glm_geometry_rejects_model_drift(monkeypatch):
    case = _case()
    case["reference_profile"] = "fixed"
    manifest = {"platform": "a2a3", "sampling": {"rounds": 100, "warmup": 10},
                "reference_profiles": {"fixed": {"model_config": {"hidden_size": 4096},
                                                 "tp": 16, "ep": 16}}}
    config = SimpleNamespace(FLASH=SimpleNamespace(hidden_size=4096), TP_SIZE=16, EP_SIZE=16)
    monkeypatch.setattr(perf.importlib, "import_module", lambda name: config)
    perf.validate_glm_dimensions(manifest, case)
    config.FLASH.hidden_size = 8192
    with pytest.raises(ValueError, match="fixed GLM parameter hidden_size changed"):
        perf.validate_glm_dimensions(manifest, case)
    config.FLASH.hidden_size = 4096
    config.TP_SIZE = 8
    with pytest.raises(ValueError, match="fixed GLM tp"):
        perf.validate_glm_dimensions(manifest, case)


def test_heap_override_survives_prepare_and_allocated_capture(prepared_environment, tmp_path, monkeypatch):
    manifest, case, _, _ = prepared_environment
    case["runtime_config"] = {"ring_heap": 2 << 30}
    golden = sys.modules["golden"]
    original = golden.run
    configs = []
    def record(**kwargs):
        configs.append(dict(kwargs["config"]))
        return original(**kwargs)
    monkeypatch.setattr(golden, "run", record)
    path = tmp_path / "case/prepared.json"
    perf.prepare_case(SimpleNamespace(case="glm", output=path))
    perf.replay_prepared(manifest, case, [7], tmp_path / "result.json", path)
    assert len(configs) == 2
    assert all(config["ring_heap"] == 2 << 30 for config in configs)
    assert all(config["ring_task_window"] == perf.RING_CONFIG["ring_task_window"] for config in configs)
    assert configs[-1]["device_id"] == 7
    assert json.loads((tmp_path / "result.json").read_text())["runtime_config"]["ring_heap"] == 2 << 30


@pytest.mark.parametrize("overrides", [
    {"device_id": 7}, {"backend_type": "a5"}, {"ring_heap": 0},
    {"ring_heap": -1}, {"ring_heap": True}, {"ring_heap": 2.0}, [],
])
def test_runtime_overrides_reject_device_and_invalid_ring_settings(overrides):
    with pytest.raises(ValueError, match="runtime_config"):
        perf.ring_config({"runtime_config": overrides})


def test_legacy_capture_keeps_default_ring_settings(tmp_path):
    seen = {}
    def run(**kwargs):
        seen.update(kwargs["config"])
        return SimpleNamespace(passed=False, error="expected diagnostic")
    perf.make_capture(run, tmp_path / "result.json", [3])(
        config={"device_id": 3, "ring_heap": 64},
    )
    assert seen["ring_heap"] == 1 << 30
    assert seen["device_id"] == 3
