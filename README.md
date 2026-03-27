# AgierCompute - 跨平台 GPU 计算框架

一个让用户**轻松编写跨 GPU 内核**的计算框架。

## 核心理念

**不提供预置内核**，只提供让用户方便编写自己内核的工具。

## 快速开始

### 1. 定义内核

```c
#include "ace.h"

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### 2. 选择设备

```c
ace_device_t dev;
ace_device_get(ACE_DEVICE_CUDA, 0, &dev);  // 或 OPENCL/VULKAN
```

### 3. 执行内核

#### 方式一：使用 ACE_INVOKE 宏（推荐）

```c
const int N = 1000;
float a[N], b[N], c[N];

ace_buffer_t ba, bb, bc;
ace_buffer_alloc(dev, N * sizeof(float), &ba);
ace_buffer_alloc(dev, N * sizeof(float), &bb);
ace_buffer_alloc(dev, N * sizeof(float), &bc);

ace_buffer_write(ba, a, N * sizeof(float));
ace_buffer_write(bb, b, N * sizeof(float));

/* 使用 ACE_INVOKE 宏 - 自动处理参数类型 */
int n = N;  /* 注意：标量参数必须是变量，不能是字面量 */
ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, n, ba, bb, bc);
ace_finish(dev);

ace_buffer_read(bc, c, N * sizeof(float));
ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
ace_device_release(dev);
```

#### 方式二：使用原始 API

```c
int n = N;
void* args[] = {&n, ba, bb, bc};
int sizes[] = {sizeof(int), 0, 0, 0};  /* 0 表示 buffer */

ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, sizes, 4);
ace_finish(dev);
```

> **重要提示**：`ACE_INVOKE` 宏的标量参数必须是**左值（变量）**，不能是字面量或表达式。
> - ✅ 正确：`int n = 100; ACE_INVOKE(..., n, ...);`
> - ❌ 错误：`ACE_INVOKE(..., 100, ...);`
> - ❌ 错误：`ACE_INVOKE(..., N + 1, ...);`
> 
> 如需传递字面量或表达式，请先赋值给局部变量。

## 支持的后端

| 后端 | 状态 | 编译方式 | 说明 |
|------|------|----------|------|
| CUDA | ✅ | NVRTC | 表驱动架构 |
| OpenCL | ✅ | 运行时 | 表驱动架构 |
| Vulkan | ✅ | shaderc | 表驱动架构 |
| CPU | ❌ | - | 占位实现 |
| Metal | ❌ | - | 未实现 |

> 使用 `ACE_DEVICE_ALL` 可遍历所有可用设备。详见 [API 文档](docs/API.md)

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 依赖

- **CUDA**: NVIDIA CUDA Toolkit
- **OpenCL**: OpenCL SDK
- **Vulkan**: Vulkan SDK + shaderc

## 运行测试

```bash
cd build
ctest --output-on-failure      # 所有测试
ctest -L Backend               # 后端测试
ctest -R CUDA_Backend          # 单个测试
```

### 测试状态

✅ **所有测试通过 (9/9)**

- CUDA_Backend
- OpenCL_Backend
- Vulkan_Backend
- All_Kernels
- Data_Types
- All_DataTypes
- Comprehensive_Test (255/255)
- Example_UserKernels
- Demo_CPP

## 示例程序

| 示例 | 说明 |
|------|------|
| `simple_test` | 基础测试 |
| `test_cuda` | CUDA 后端测试 |
| `test_opencl` | OpenCL 后端测试 |
| `test_vulkan` | Vulkan 后端测试 |
| `benchmark` | 性能基准测试 |
| `device_enum` | 设备遍历示例 |
| `comprehensive_test` | 综合测试 (3 后端×9 类型×9 操作) |

## 内核内建变量

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 屏障同步 |

## 架构

### 表驱动设计

```
┌─────────────────────────────────────────────────────┐
│                    用户应用层                        │
│              ACE_KERNEL, ace_kernel_invoke          │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│                    核心引擎层                        │
│        设备管理、内存管理、内核调度、参数传递        │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│                   后端实现层                         │
│  ┌──────────┬──────────┬──────────┬─────────────┐  │
│  │   CUDA   │  OpenCL  │  Vulkan  │    CPU      │  │
│  │ dtype_   │ dtype_   │ dtype_   │ (placeholder)│  │
│  │ table    │ table    │ table    │             │  │
│  │ ops_     │ ops_     │ ops_     │             │  │
│  │ table    │ table    │ table    │             │  │
│  └──────────┴──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────────────────┘
```

**架构优势：**
- **高内聚**: 每个操作的完整实现在一个 inject 函数内
- **低耦合**: 类型和操作完全隔离
- **易扩展**: 添加新类型只需在类型表中添加条目
- **易维护**: 所有类型/操作相关逻辑集中在表中

详见 [后端状态](docs/BACKEND_STATUS.md)。

## 数据类型支持

| 数据类型 | 大小 | CUDA | OpenCL | Vulkan | 说明 |
|----------|------|------|--------|--------|------|
| **FLOAT32** | 4B | ✅ | ✅ | ✅ | 单精度浮点 |
| **FLOAT64** | 8B | ✅ | ✅ | ✅ | 双精度浮点 |
| **INT32** | 4B | ✅ | ✅ | ✅ | 32 位整数 |
| **INT64** | 8B | ✅ | ✅ | ✅ | 64 位整数 |
| **FLOAT16** | 2B | ✅ | ✅ | ✅ | 半精度浮点 (AI) |
| **BFLOAT16** | 2B | ✅ | ✅ | ✅ | Brain 浮点 (AI) |
| **INT8** | 1B | ✅ | ✅ | ✅ | 8 位整数 (量化) |
| **UINT8** | 1B | ✅ | ✅ | ✅ | 8 位无符号整数 |
| **INT16** | 2B | ✅ | ✅ | ✅ | 16 位整数 |

## 文档

- [API 参考](docs/API.md) - 完整 API 文档
- [后端状态](docs/BACKEND_STATUS.md) - 各后端实现状态和测试报告

## 许可证

MIT License
