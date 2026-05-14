# CMakeLists.txt 模板

> **使用方式**：复制此模板到 `{operator_name}/CMakeLists.txt`，修改 `{operator_name}` 和 `{NPU架构}`

---

## 标准模板（Kernel 直调）

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project({operator_name}_custom LANGUAGES ASC CXX)

add_executable({operator_name}_custom
    {operator_name}.asc
)

target_include_directories({operator_name}_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries({operator_name}_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options({operator_name}_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch={NPU架构}>
)
```

---

## 需要修改的字段

| 字段 | 说明 | 示例 |
|-----|------|------|
| `{operator_name}` | 算子名称（小写） | `softmax0309` |
| `{NPU架构}` | NPU 架构代码 | `dav-2201` (A2/A3) / `dav-2002` (推理) |

---

## NPU 架构对照表

| NPU 型号 | `--npu-arch` |
|---------|-------------|
| Atlas A3 系列 (910B3) | `dav-2201` |
| Atlas A2 系列 | `dav-2201` |
| Atlas 推理系列 | `dav-2002` |
| Atlas 训练系列 (910A) | `dav-1001` |

---

## 关键配置说明

### 1. 语言声明（必须）

```cmake
# ✅ 正确
project(<operator_name> LANGUAGES ASC CXX)

# ❌ 错误 - 缺少 ASC 语言声明
project(<operator_name> LANGUAGES CXX)
```

**错误后果**：
```
CMake Error: Cannot determine link language for target "xxx".
```

### 2. 必需链接库

| 库名 | 用途 |
|-----|------|
| `tiling_api` | Tiling API 支持 |
| `register` | 寄存器操作 |
| `platform` | 平台相关 |
| `unified_dlog` | 日志 |
| `dl` | 动态链接 |
| `m` | 数学库 |
| `graph_base` | 图基础 |

**缺少链接库的错误**：
```
undefined reference to `AscendC::xxx'
```

### 3. 头文件包含（多文件算子）

```cmake
target_include_directories(<operator_name> PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

---

## 常见错误

### 错误1：缺少 ASC 语言声明

**错误信息**：
```
CMake Error: Cannot determine link language for target "xxx".
```

**解决方案**：
```cmake
project(xxx LANGUAGES ASC CXX)  # 添加 ASC
```

### 错误2：npu-arch 配置错误

**错误信息**：
```
error: unknown target architecture 'xxx'
```

**解决方案**：根据 NPU 型号选择正确的 `--npu-arch` 参数（见上表）

### 错误3：缺少必需链接库

**错误信息**：
```
undefined reference to `AscendC::xxx'
```

**解决方案**：确保链接所有必需库（见上表）

### 错误4：头文件找不到

**错误信息**：
```
fatal error: xxx.h: No such file or directory
```

**解决方案**：
```cmake
target_include_directories(<operator_name> PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

---

## 验证命令

```bash
python <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_cmake_config.py \
    ops/{operator_name}/CMakeLists.txt
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

---

## 构建流程

```bash
# 1. 创建构建目录
mkdir build && cd build

# 2. 配置 CMake
cmake ..

# 3. 编译
make

# 4. 运行
./<operator_name>
```

---

## 环境要求

- **CMake 版本**：≥ 3.16
- **CANN 版本**：≥ 9.0.0
- **编译器**：bisheng（CANN 自带）
- **链接器**：ld.lld（CANN 自带）
- **环境变量**：`ASCEND_HOME_PATH` 必须设置

---

## 示例

**softmax0309 算子（A3 服务器）**：

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(softmax0309_custom LANGUAGES ASC CXX)

add_executable(softmax0309_custom
    softmax0309.asc
)

target_include_directories(softmax0309_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(softmax0309_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options(softmax0309_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

---

## 需要修改的字段

| 字段 | 说明 | 示例 |
|-----|------|------|
| `{operator_name}` | 算子名称（小写） | `softmax0309` |
| `{NPU架构}` | NPU 架构代码 | `dav-2201` (A2/A3) / `dav-2002` (推理) |

---

## NPU 架构对照表

| NPU 型号 | `--npu-arch` |
|---------|-------------|
| Atlas A3 系列 (910B3) | `dav-2201` |
| Atlas A2 系列 | `dav-2201` |
| Atlas 推理系列 | `dav-2002` |
| Atlas 训练系列 (910A) | `dav-1001` |

---

## 验证命令

```bash
python <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_cmake_config.py \
    ops/{operator_name}/CMakeLists.txt
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

---

## 示例

**softmax0309 算子（A3 服务器）**：

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(softmax0309_custom LANGUAGES ASC CXX)

add_executable(softmax0309_custom
    softmax0309.asc
)

target_include_directories(softmax0309_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(softmax0309_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options(softmax0309_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```
