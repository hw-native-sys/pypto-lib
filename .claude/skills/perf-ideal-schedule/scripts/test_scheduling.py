# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Behavioral regression tests; run directly with OR-Tools on PYTHONPATH."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze
from scheduling import expand_waves, graph_times, list_schedule, pressure_report, verify_schedule


def task(tid, blocks=1, kernels=(-1, 0, -1), weight=1):
    return (
        {"task_id": tid, "block_num": blocks, "kernel_ids": list(kernels)},
        {"task_id": tid, "blocks": blocks, "name": "kernel_" + tid, "max_us": weight},
    )


def prepare(tasks, edges=()):
    deps = {"tasks": [t[0] for t in tasks], "edges": [{"pred": a, "succ": b} for a, b in edges]}
    info, stats, expanded_edges, waves = expand_waves(deps, [t[1] for t in tasks])
    graph = graph_times(info, stats, expanded_edges)
    schedule = list_schedule(info, stats, graph)
    verify_schedule(info, graph[0], expanded_edges, *schedule)
    return info, stats, expanded_edges, waves, graph, schedule


class SchedulingTests(unittest.TestCase):
    def test_oversized_work_and_dependencies(self):
        tasks = [
            task("producer", weight=2),
            task("reduce", 192),
            task("alloc", weight=0),
            task("consumer", weight=3),
            task("side", kernels=(0, -1, -1), weight=4),
        ]
        info, stats, edges, waves, graph, schedule = prepare(
            tasks, [("producer", "reduce"), ("reduce", "alloc"), ("alloc", "consumer")]
        )
        self.assertEqual(waves[0]["wave_blocks"], [64, 64, 64])
        self.assertEqual(sum(info[w]["block_num"] for w in waves[0]["wave_ids"]), 192)
        self.assertEqual(max(schedule[1].values()), 8000)
        self.assertGreaterEqual(schedule[0]["consumer"], schedule[1]["reduce:wave2"])
        self.assertEqual(tasks[1][0]["block_num"], 192)
        self.assertEqual(tasks[1][1]["max_us"], 1)

    def test_partial_wave_and_mix_slots(self):
        for kernels, blocks, expected in [
            ((0, -1, -1), 33, [32, 1]),
            ((-1, 0, -1), 65, [64, 1]),
            ((-1, 0, 1), 65, [32, 32, 1]),
            ((0, 1, 2), 65, [32, 32, 1]),
            ((0, -1, 2), 33, [32, 1]),
        ]:
            with self.subTest(kernels=kernels):
                info, stats, edges, waves, graph, schedule = prepare([task("x", blocks, kernels)])
                self.assertEqual(waves[0]["wave_blocks"], expected)
                self.assertEqual(max(schedule[1].values()), len(expected) * 1000)

    def test_excluded_task_keeps_edges(self):
        _, _, edges, waves, _, schedule = prepare(
            [task("a", weight=2), task("excluded", 192, weight=0), task("b")],
            [("a", "excluded"), ("excluded", "b")],
        )
        self.assertFalse(waves)
        self.assertEqual(len(edges), 2)
        self.assertEqual(schedule[0]["b"], 2000)

    def test_pressure_uses_core_demand_not_task_count(self):
        info, stats, _, _, graph, schedule = prepare([task("a", 40), task("b", 40)])
        pressure = pressure_report(info, stats, graph[3], graph[4])
        peak = pressure["peaks"][0]
        self.assertEqual((peak["parallel_tasks"], peak["peak_demand"], peak["capacity"]), (2, 80, 64))
        self.assertEqual(max(schedule[1].values()), 2000)
        self.assertEqual(pressure["overloaded_us"], 1)
        self.assertEqual(pressure["top_windows"][0]["excess_core_us"], 16)
        self.assertEqual(pressure["top_windows"][0]["end_us"], 1)
        info, stats, _, _, graph, _ = prepare([task("a", 64), task("b", 64)], [("a", "b")])
        self.assertFalse(pressure_report(info, stats, graph[3], graph[4])["peaks"])

    def test_cycle_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            prepare([task("a"), task("b")], [("a", "b"), ("b", "a")])

    def test_independent_verifier_rejects_wrong_mix_slot(self):
        info, _, edges, _, graph, schedule = prepare([task("a", kernels=(0, -1, 2))])
        starts, ends, assigned = schedule
        assigned["a"] = [0, 32]
        with self.assertRaisesRegex(ValueError, "MIX pairing"):
            verify_schedule(info, graph[0], edges, starts, ends, assigned)

    def test_large_parallel_graph(self):
        info, stats, _, _, graph, schedule = prepare([task(str(i)) for i in range(10000)])
        self.assertEqual(max(schedule[1].values()), 157000)
        self.assertEqual(pressure_report(info, stats, graph[3], graph[4])["peaks"][0]["peak_demand"], 10000)

    def run_case(self, **options):
        tasks = [task("reduce", 192), task("tail", 32, (0, 1, 2))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chip_swimlane_records.json"
            (path.parent / "deps.json").write_text(
                json.dumps({"tasks": [t[0] for t in tasks], "edges": [{"pred": "reduce", "succ": "tail"}]})
            )
            return analyze.run(
                "test",
                {
                    "rank": "r",
                    "ranks": {"r": {"records": str(path), "tasks": [t[1] for t in tasks], "kernel_us": 2}},
                    "timeout": 5,
                    **options,
                },
            )

    def test_exact_and_auto_limit(self):
        exact = self.run_case()
        self.assertEqual(exact["resource_schedule_us"], 4)
        self.assertEqual(exact["previous_path_us"], 4)
        self.assertEqual(exact["single_wave_dependency_path_us"], 2)
        self.assertEqual(exact["status"], "OPTIMAL")
        self.assertEqual(exact["resource_lower_bound_us"], 4)
        with patch.object(analyze, "solve_exact", side_effect=AssertionError("must skip CP-SAT")):
            heuristic = self.run_case(exact_task_limit=1)
        self.assertEqual(heuristic["status"], "HEURISTIC")
        self.assertEqual(heuristic["resource_schedule_us"], 4)
        self.assertIsNone(heuristic["engine_schedule_us"])

    def test_engine_timeout_keeps_verified_fallback(self):
        with (
            patch.object(analyze.cp_model.CpSolver, "solve", return_value=analyze.cp_model.UNKNOWN),
            patch.object(analyze.cp_model.CpSolver, "best_objective_bound", 0),
        ):
            result = self.run_case()
        self.assertEqual(result["status"], "HEURISTIC")
        self.assertEqual(result["resource_schedule_us"], 4)
        self.assertEqual(result["engine_status"], "UNKNOWN")
        self.assertTrue(result["physical_cores"])

    def test_placement_timeout_keeps_engine_diagnostic(self):
        original = analyze.cp_model.CpSolver.solve
        calls = 0

        def solve(solver, model):
            nonlocal calls
            calls += 1
            return original(solver, model) if calls == 1 else analyze.cp_model.UNKNOWN

        with patch.object(analyze.cp_model.CpSolver, "solve", solve):
            result = self.run_case()
        self.assertEqual(result["status"], "HEURISTIC")
        self.assertEqual(result["engine_status"], "OPTIMAL")
        self.assertEqual(result["paired_core_assignment"], "UNKNOWN")
        self.assertEqual(result["engine_schedule_us"], 4)
        self.assertEqual(result["resource_schedule_us"], 4)


if __name__ == "__main__":
    unittest.main()
