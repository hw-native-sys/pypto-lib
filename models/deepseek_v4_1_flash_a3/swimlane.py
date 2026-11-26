# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Archive and attribute the 16-rank A3 Level 1 capture."""

import csv
import json
import re
from collections import Counter
from statistics import mean

# A fused task belongs to one category. Module-specific kernels take precedence
# over their constituent instructions (e.g. Indexer scores also use matmul).
CATEGORY_PATTERNS = {
    "mHC Pre": r"^(mhc_|hc_collapse)",
    "mHC Post": r"^(hc_post|sp_mhc_post)",
    "Attention core": r"^(swa_attn|c[12]a_sparse_attn|sparse_online|sparse_bf16_attn)",
    "Indexer": (r"(^|_)(idx|index|indexer)(_|$)|^((prefill|c1a)_pool_|pool_(all|score|select|count|topk)"
                r"|prefill_(inactive_pool|c2a_topk)|c1a_mx_projection)"),
    "MoE": r"^(shared_|routed_(gate|down|swiglu))",
    "MoE routing": (r"^(gate(_|$)|route_sort|dispatch_|combine_|routed_return|moe_reuse_wait|moe_signal"
                    r"|prefill_moe_(compact|scatter|counts)|moe_round_(release|counts|rows))"),
    "Matmul / GEMM": r"^(proj_|engram_projection|lm_head_matmul)|(^|_)(matmul|gemm)(_|$)",
    "Norm / RoPE / quantization / activation": (r"norm|rope|quant|swiglu|^swa_decode_cache"
                                                r"|^sparse_shared_(window|compressed)_decode"),
    "Memory copy / fill": (r"(^|_)(copy|fill|zero|padding|pack|gather)(_|$)|^prefill_swa_(stage|collect)"
                           r"|^swa_stage_runs|^lm_head_(dispatch|combine)_push"),
}
OTHER = "Other"
CATEGORIES = (*CATEGORY_PATTERNS, OTHER)


def step_name(step):
    return "prefill" if step == 0 else f"decode-{step:02d}"


def profile_program_dirs(directory, artifacts):
    """Give both programs private DFX scratch while reusing the compiled files."""
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    for mode, artifact in zip(("prefill", "decode"), artifacts, strict=True):
        view = directory / ".programs" / mode
        view.mkdir(parents=True)
        for name in ("distributed_meta.json", "orchestration", "next_levels", "sub_workers"):
            source = artifact.resolve() / name
            if source.exists():
                (view / name).symlink_to(source)
        paths.append(view)
    return paths


def kernel_category(kernel, program):
    for category, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, kernel):
            return category
    if re.fullmatch(r"l2_(prefill|decode)_fwd_incore_\d+", kernel):
        source = program / "kernels" / "aic" / f"{kernel}.cpp"
        if source.is_file() and "TMATMUL(" in source.read_text():
            return "Matmul / GEMM"
    return OTHER


def dispatch_totals(records, program, kernel_categories):
    """Count logical task spans and physical core execution cycles separately."""
    directory = records.parent
    data = json.loads(records.read_bytes())
    metadata = data["metadata"]
    frequency = metadata["clock_freq_hz"]
    collection = metadata.get("collection", {})
    if (data["chip_swimlane_level"] != 1 or not data["aicore_tasks"] or frequency <= 0
            or collection.get("verdict", "published") != "published"):
        raise ValueError(f"invalid or incomplete Level 1 capture: {records}")
    core_types = metadata["core_types"]
    deps = directory / "deps.json"
    if not deps.is_file():
        raise ValueError(f"missing task graph: {deps}")
    names = json.loads((directory / "name_map.json").read_bytes())["callable_id_to_name"]
    categories = {key: kernel_category(name, program) for key, name in names.items()}
    kernel_categories.update({name: categories[key] for key, name in names.items()})
    # A3 submits block_num blocks for every active subslot. Dummy tasks
    # have no active subslots. This model does not use dispatch predicates.
    lanes = {}
    remaining = Counter()
    for entry in json.loads(deps.read_bytes())["tasks"]:
        task = int(entry["task_id"])
        aic, aiv0, aiv1 = entry["kernel_ids"]
        for kind, ids in (("aic", (aic,)), ("aiv", (aiv0, aiv1))):
            active = [categories[str(i)] for i in ids if i >= 0]
            if active:
                # Level 1 distinguishes AIC from AIV, but not AIV0 from AIV1.
                lanes[task, kind] = active[0] if len(set(active)) == 1 else OTHER
                remaining[task, kind] = int(entry["block_num"]) * len(active)
    spans = {}
    busy = Counter()
    for core, task, _, start, end, _, epoch in data["aicore_tasks"]:
        if not 0 <= core < len(core_types) or not 0 < start <= end:
            raise ValueError(f"invalid core or task timestamps in {records}")
        lane = task, core_types[core]
        if lane not in lanes:
            raise ValueError(f"unresolved task {task} on core {core} in {records}")
        remaining[lane] -= 1
        category = lanes[lane]
        busy[category] += end - start
        key = epoch, task
        if key in spans:
            first, last, prior = spans[key]
            spans[key] = min(first, start), max(last, end), prior if prior == category else OTHER
        else:
            spans[key] = start, end, category
    for lane, count in remaining.items():
        if count:
            raise ValueError(f"graph/timing block count mismatch for {lane} in {records}: difference {count}")
    elapsed = Counter()
    for first, last, category in spans.values():
        elapsed[category] += last - first
    ms_per_tick = 1000 / frequency
    return {cat: (elapsed[cat] * ms_per_tick, busy[cat] * ms_per_tick) for cat in CATEGORIES}


def summarize_swimlanes(directory, devices, owner=0, *, convert=False):
    """Publish timing tables only after validating all eight steps on every rank."""
    by_rank = [[] for _ in devices]
    kernel_categories = {}
    name_maps = {}
    if convert:
        from pypto.runtime.distributed_runner import _write_dispatch_name_map
        from pypto.runtime.runner import _generate_swimlane
    for step in range(8):
        capture = directory / step_name(step)
        mode = "decode" if step else "prefill"
        program = directory / ".programs" / mode / "next_levels" / f"l2_{mode}_fwd"
        for rank, samples in enumerate(by_rank):
            # This model submits one complete L2 graph per rank per forward.
            files = list((capture / f"rank{rank}").glob("d[0-9]*/chip_swimlane_records.json"))
            if len(files) != 1:
                raise ValueError(f"expected one Level 1 capture in {capture}/rank{rank}, found {len(files)}")
            records = files[0]
            if convert:
                dispatch = records.parent
                deps = dispatch / "deps.json"
                if not deps.is_file():
                    raise ValueError(f"missing same-dispatch task graph: {deps}")
                names = _write_dispatch_name_map(dispatch, program, name_maps)
                if names is None:
                    raise ValueError(f"cannot resolve kernel names for {program}")
            samples.append(dispatch_totals(records, program, kernel_categories))
            if convert:
                _generate_swimlane(program, dispatch, records, func_names=names, deps_json=deps)
    summary = []
    with (directory / "per_rank.csv").open("w", newline="") as stream:
        output = csv.writer(stream)
        output.writerow(["rank", "device", "category", "prefill_task_ms",
                         *(f"{step_name(i)}_task_ms" for i in range(1, 8)), "decode_mean_task_ms",
                         "prefill_core_ms", "decode_mean_core_ms"])
        for rank, steps in enumerate(by_rank):
            for category in CATEGORIES:
                task_ms, core_ms = zip(*(step[category] for step in steps))
                row = dict(category=category, prefill_task_ms=task_ms[0], decode_mean_task_ms=mean(task_ms[1:]),
                           prefill_core_ms=core_ms[0], decode_mean_core_ms=mean(core_ms[1:]))
                output.writerow([rank, devices[rank], category, *task_ms, row["decode_mean_task_ms"],
                                 core_ms[0], row["decode_mean_core_ms"]])
                if rank == owner:
                    summary.append(row)
    with (directory / "operator_summary.csv").open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(summary[0]))
        output.writeheader()
        output.writerows(summary)
    (directory / "kernel_categories.json").write_text(json.dumps(kernel_categories, indent=2, sort_keys=True) + "\n")
    return dict(directory=str(directory.resolve()), rank=owner, device=devices[owner], level=1, categories=summary)
