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

```c
const int N = 1000;
float a[N], b[N], c[N];

ace_buffer_t ba, bb, bc;
ace_buffer_alloc(dev, N * sizeof(float), &ba);
ace_buffer_alloc(dev, N * sizeof(float), &bb);
ace_buffer_alloc(dev, N * sizeof(float), &bc);

ace_buffer_write(ba, a, N * sizeof(float));
ace_buffer_write(bb, b, N * sizeof(float));

int n = N;
void* args[] = {&n, ba, bb, bc};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
ace_finish(dev);

ace_buffer_read(bc, c, N * sizeof(float));

ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
ace_device_release(dev);
```

## 支持的后端

| 后端 | 状态 | 编译方式 | 说明 |
|------|------|----------|------|
| CUDA | ✅ | NVRTC | NVIDIA GPU 运行时编译 |
| OpenCL | ✅ | 运行时 | 跨平台 GPU/CPU |
| Vulkan | ✅ | shaderc | SPIR-V 编译 |
| CPU | ❌ | - | 未实现 |
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

## 示例程序

| 示例 | 说明 |
|------|------|
| `simple_test` | 基础测试 |
| `test_cuda` | CUDA 后端测试 |
| `test_opencl` | OpenCL 后端测试 |
| `test_vulkan` | Vulkan 后端测试 |
| `benchmark` | 性能基准测试 |
| `device_enum` | 设备遍历示例 |

## 内核内建变量

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 屏障同步 |

## 架构

```
┌─────────────────┐
│   用户应用层     │  ACE_KERNEL, ace_kernel_invoke
└────────┬────────┘
         │
┌────────▼────────┐
│   核心引擎层     │  设备管理、内存管理、内核调度
└────────┬────────┘
         │
┌────────┼────────┬────────┬────────┐
│  CUDA  │ OpenCL │ Vulkan │  CPU   │
│ NVRTC  │ Runtime│ SPIR-V │  (TODO)│
└────────┴────────┴────────┴────────┘
```

## 文档

- [API 参考](docs/API.md) - 完整 API 文档
- [后端状态](docs/BACKEND_STATUS.md) - 各后端实现状态

## 许可证

MIT License
