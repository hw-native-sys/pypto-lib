#!/usr/bin/env python3
"""
计算每个 InCore 函数中「局部变量」（pl.Tensor 类型）的总大小，作为 SRAM 的预估值。
规则：只统计函数体内赋值的 tensor（不含参数）；按 MemRef id 去重后求和；
MemRef 中 size=0 的 view 按 shape×dtype 计算逻辑大小。
"""
import re
from pathlib import Path
from collections import defaultdict

DUMP = Path(__file__).resolve().parent.parent / "passes_dump" / "13_after_AllocateMemoryAddr.py"
DTYPE_BYTES = {"pl.BFLOAT16": 2, "pl.FP32": 4, "pl.INT32": 4}


def parse_shape(shape_str: str) -> int:
    """e.g. '16, 128' -> 16*128; '1, 128 // 2' -> 64."""
    prod = 1
    for part in shape_str.split(","):
        part = part.strip().replace(" ", "")
        if "//" in part:
            a, b = part.split("//")
            prod *= int(a) // int(b)
        else:
            prod *= int(part)
    return prod


def main():
    lines = DUMP.read_text().splitlines()
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 只处理 InCore 的 def
        if re.match(r"    def qwen3_decode_layer_incore", line):
            name = re.search(r"def (qwen3_decode_layer_incore[\w]+)\(", line).group(1)
            # 跳过 Orchestration / function_group（看上一行）
            if i > 0 and "Orchestration" in lines[i - 2]:
                i += 1
                continue
            if i > 0 and "function_group" in lines[i - 2]:
                i += 1
                continue
            # 参数 MemRef id
            param_ids = set(int(x) for x in re.findall(r"MemRef\([^)]+,\s*(\d+)\s*\)", line))
            id_to_size = {}
            i += 1
            while i < len(lines):
                cur = lines[i]
                # 遇到下一处顶层 def 或 @pl.function / @pl.function_group 则结束
                if re.match(r"    (def |@pl\.function)", cur):
                    break
                if "pl.Tensor" in cur and "=" in cur:
                    mem = re.search(r"pl\.MemRef\([^,]+,\s*-?\d+,\s*(\d+),\s*(\d+)\s*\)", cur)
                    if mem:
                        size = int(mem.group(1))
                        mid = int(mem.group(2))
                        if mid not in param_ids:
                            if size == 0:
                                shape_m = re.search(r"pl\.Tensor\[\[([^\]]+)\]\]", cur)
                                dtype_m = re.search(r"pl\.(BFLOAT16|FP32|INT32)", cur)
                                if shape_m and dtype_m:
                                    try:
                                        numel = parse_shape(shape_m.group(1))
                                    except Exception:
                                        numel = 1
                                    size = numel * DTYPE_BYTES.get("pl." + dtype_m.group(1), 4)
                            # 排除超大 buffer（通常为 assemble 写回参数，不会进 SRAM）
                            if size <= 1024 * 1024:  # 1 MB
                                id_to_size[mid] = size
                i += 1
            total = sum(id_to_size.values())
            results.append((name, total, len(id_to_size)))
        else:
            i += 1
    # 输出：函数级
    print("InCore 函数 | 局部 tensor 总大小 (B) | 去重 buffer 数")
    print("-" * 65)
    for name, total, n in sorted(results, key=lambda x: -x[1]):
        print(f"{name:48} | {total:>8} | {n}")
    total_all = sum(r[1] for r in results)
    print("-" * 65)
    print(f"{'合计 (所有 InCore 局部 tensor 去重后总和)':48} | {total_all:>8}")

    # 输出：function_group 级（AIC/AIV 并列，不合并求和）
    group_split = defaultdict(lambda: {"aic": 0, "aiv": 0, "solo": 0})
    for name, total, _ in results:
        m = re.match(r"(.+)_(aic|aiv)$", name)
        if m:
            gname, kind = m.group(1), m.group(2)
            group_split[gname][kind] = total
        else:
            group_split[name]["solo"] = total

    print()
    print("function_group(逻辑 kernel) | AIC 局部 tensor(B) | AIV 局部 tensor(B) | 单函数(B)")
    print("-" * 96)
    sorted_groups = sorted(
        group_split.items(),
        key=lambda kv: max(kv[1]["aic"], kv[1]["aiv"], kv[1]["solo"]),
        reverse=True,
    )
    for gname, data in sorted_groups:
        print(f"{gname:32} | {data['aic']:>16} | {data['aiv']:>16} | {data['solo']:>9}")
    print("-" * 96)
    print(
        f"{'合计（分列）':32} | "
        f"{sum(x['aic'] for x in group_split.values()):>16} | "
        f"{sum(x['aiv'] for x in group_split.values()):>16} | "
        f"{sum(x['solo'] for x in group_split.values()):>9}"
    )
    return results


if __name__ == "__main__":
    main()
