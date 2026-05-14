---
name: ascendc-docs-search
description: Ascend C 开发资源索引（本地+在线）。提供：(1) 本地 API 文档索引、示例代码映射，(2) 在线文档搜索功能，(3) 资源查找优先级，(4) Explore Agent 使用指南。优先使用本地资源，仅在本地检索不到时使用在线搜索。
---

# Ascend C 开发资源

## 概述

本技能提供"本地优先，在线兜底"的文档搜索能力：
- **本地资源**：1022 个 API 文档、587 个示例代码、实现参考
- **在线搜索**：华为昇腾社区文档搜索（仅在本地资源不足时使用）

## 官方资源路径

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | `asc-devkit/docs/api/context/` | 1022 个 API 文档 |
| 高性能模板 | `asc-devkit/examples/00_introduction/01_add/basic_api_memory_allocator_add/` | 双缓冲+流水线标准实现 |
| 各类示例 | `asc-devkit/examples/00_introduction/` | 加法、减法、多输入等 |
| 完整文档 | `asc-devkit/docs/` | 完整开发文档 |
| Tiling 实现 | `asc-devkit/impl/adv_api/tiling/` | Tiling 参数配置参考 |
| 矢量计算 | `asc-devkit/examples/00_introduction/11_vectoradd/` | 矢量 API 使用 |
| 调试示例 | `asc-devkit/examples/01_utilities/00_printf/printf.asc` | printf 调试方法 |

## 资料查找优先级

```
1. asc-devkit/docs/api/context/ (本地 API 文档 - 1022 个)
         ↓ 找不到
2. asc-devkit/examples/ (示例代码 - 587 个)
         ↓ 找不到
3. asc-devkit/impl/ (实现代码)
         ↓ 找不到
4. 在线搜索（华为昇腾社区）
   使用 scripts/ 中的 Python 脚本
   推荐版本：9.0.0（与当前环境一致）
```

## 环境兼容性

**当前环境**：A3 服务器，CANN 9.0.0

查阅资料时必须确认 API/方法适用于当前环境。

## Explore Agent 使用

**何时使用**：
1. 查找 API 文档或实现示例
2. 了解某个技术点的实现方式
3. 搜索类似算子的参考实现
4. 遇到错误时查找相关解决方案
5. 方案走通后探索更优实现

**使用方式**：
```
使用 Task 工具，subagent_type=Explore
提供明确的搜索目标和范围
```

**示例提示词**：
- "搜索 asc-devkit 中 Exp API 的使用示例"
- "查找双缓冲的实现参考"
- "搜索类似的三角函数算子实现"

## 示例代码索引

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| 高性能模板 | `asc-devkit/examples/00_introduction/01_add/basic_api_memory_allocator_add/` | 双缓冲+流水线 |
| 多输入加法 | `asc-devkit/examples/00_introduction/04_addn/addn.asc` | 多输入处理 |
| 减法算子 | `asc-devkit/examples/00_introduction/07_sub/sub_custom.asc` | 减法实现 |
| 调试打印 | `asc-devkit/examples/01_utilities/00_printf/printf.asc` | printf 调试 |
| 断言使用 | `asc-devkit/examples/01_utilities/01_assert/assert.asc` | 断言示例 |
| 库函数 | `asc-devkit/examples/03_libraries/00_addcdivcustom/addcdiv_custom.asc` | 库函数使用 |
| 矢量计算 | `asc-devkit/examples/00_introduction/11_vectoradd/vector_add_custom.asc` | 矢量 API |

## 在线搜索

**适用情况**：
- 本地 context 目录检索不到相关 API
- 需要更详细的官方说明或最新版本信息
- 本地文档版本过旧或不完整

**快速搜索**：
```bash
# 基础搜索（推荐使用中文关键词）
python skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --max_results 5

# 搜索 API 文档
python skills/ascendc-docs-search/scripts/ascend_search_client.py "AscendC::Add 接口原型" --max_results 8

# 带版本过滤（推荐使用 9.0.0）
python skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C API" --version "9.0.0"
```

**获取详细内容**：
```bash
python skills/ascendc-docs-search/scripts/ascend_content_fetcher.py <URL>
```

**参数说明**：
- `keyword`：搜索关键词（必需），建议使用中文
- `--max_results`：返回结果数量（1-10，默认 10）
- `--lang`：语言设置（zh/en，默认 zh）
- `--version`：版本过滤字符串（如 "9.0.0"）
- `--doc_type`：文档类型（DOC/API，默认 DOC）

**依赖安装**：
```bash
pip install -r skills/ascendc-docs-search/requirements.txt
```

## 参考资料

- [API 文档索引](references/api-index.md)
- [示例代码目录](references/example-catalog.md)
- [环境兼容性表](references/compatibility.md)
