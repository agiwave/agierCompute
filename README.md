# AgierCompute - 跨平台 GPU 计算框架

一个让用户**轻松编写跨 GPU 内核**的计算框架。

## 核心理念

**不提供预置内核**，只提供让用户方便编写自己内核的工具。

## 快速开始

### 1. 定义你的内核

```c
#include "ace.h"

// 使用 ACE_KERNEL 宏定义内核
// T 是类型占位符，会自动替换为 float/int/double
ACE_KERNEL(my_kernel,
    void my_kernel(int n, T* a, T* b, T* c) {
        int i = GID;  // 全局线程 ID
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### 2. 选择设备

```c
// 获取 CPU 设备
ace_device_t cpu;
ace_device_get(ACE_DEVICE_CPU, 0, &cpu);

// 或获取 CUDA 设备
ace_device_t cuda;
ace_device_get(ACE_DEVICE_CUDA, 0, &cuda);

// 或获取 OpenCL 设备
ace_device_t opencl;
ace_device_get(ACE_DEVICE_OPENCL, 0, &opencl);

// 或获取 Vulkan 设备
ace_device_t vulkan;
ace_device_get(ACE_DEVICE_VULKAN, 0, &vulkan);
```

### 3. 执行内核

```c
// 准备数据
const int N = 1000;
float a[N] = {1, 2, 3, ...};
float b[N] = {10, 20, 30, ...};
float c[N];

// 分配设备内存
ace_buffer_t buf_a, buf_b, buf_c;
ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

// 写入数据
ace_buffer_write(buf_a, a, N * sizeof(float));
ace_buffer_write(buf_b, b, N * sizeof(float));

// 执行内核
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
ace_kernel_invoke(dev, _ace_get_my_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 4);

// 等待完成
ace_finish(dev);

// 读取结果
ace_buffer_read(buf_c, c, N * sizeof(float));

// 清理
ace_buffer_free(buf_a);
ace_buffer_free(buf_b);
ace_buffer_free(buf_c);
ace_device_release(dev);
```

## 完整示例

```c
#include <stdio.h>
#include "ace.h"

/* 定义你的向量加法内核 */
ACE_KERNEL(my_vec_add,
    void my_vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

/* 定义你的 ReLU 激活内核 */
ACE_KERNEL(my_relu,
    void my_relu(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
);

int main() {
    /* 选择设备 - 可以任意切换 */
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);
    // 或 ace_device_get(ACE_DEVICE_CUDA, 0, &dev);
    // 或 ace_device_get(ACE_DEVICE_OPENCL, 0, &dev);
    // 或 ace_device_get(ACE_DEVICE_VULKAN, 0, &dev);
    
    /* 准备数据 */
    const int N = 100;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }
    
    /* 分配设备内存 */
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(float), &ba);
    ace_buffer_alloc(dev, N * sizeof(float), &bb);
    ace_buffer_alloc(dev, N * sizeof(float), &bc);
    
    /* 写入数据 */
    ace_buffer_write(ba, a, N * sizeof(float));
    ace_buffer_write(bb, b, N * sizeof(float));
    
    /* 执行向量加法 */
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    ace_kernel_invoke(dev, _ace_get_my_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    
    /* 读取结果 */
    ace_buffer_read(bc, c, N * sizeof(float));
    // c = {0, 3, 6, 9, 12, ...}
    
    /* 清理 */
    ace_buffer_free(ba);
    ace_buffer_free(bb);
    ace_buffer_free(bc);
    ace_device_release(dev);
    
    return 0;
}
```

## 内核内建变量

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID（工作组内） |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 工作组屏障同步 |

## 支持的后端

| 后端 | 编译方式 | 说明 |
|------|----------|------|
| CPU | GCC JIT | 运行时编译，自动多线程 |
| CUDA | NVRTC | NVIDIA GPU 运行时编译 |
| OpenCL | 运行时编译 | 跨平台 GPU/CPU |
| Vulkan | shaderc | SPIR-V 编译 |

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## 运行测试

```bash
# 快速测试（CPU）
ctest -L Quick

# 所有后端测试
ctest -L Backend

# 用户内核测试
ctest -L All
```

## API 参考

### 核心 API（仅 10 个函数）

| 函数 | 描述 |
|------|------|
| `ace_device_get(type, idx, &dev)` | 获取设备 |
| `ace_device_release(dev)` | 释放设备 |
| `ace_device_count(type, &count)` | 获取设备数量 |
| `ace_device_props(dev, &props)` | 获取设备属性 |
| `ace_buffer_alloc(dev, size, &buf)` | 分配设备内存 |
| `ace_buffer_free(buf)` | 释放设备内存 |
| `ace_buffer_write(buf, data, size)` | 写入数据 |
| `ace_buffer_read(buf, data, size)` | 读取数据 |
| `ace_kernel_invoke(dev, kernel, dtype, n, args, types, nargs)` | 执行内核 |
| `ace_finish(dev)` | 等待完成 |

### 内核定义宏

```c
ACE_KERNEL(name, code)
```

- `name`: 内核名称
- `code`: 内核代码（使用 T 作为类型占位符）

### 数据类型

| 类型 | 描述 |
|------|------|
| `ACE_DTYPE_FLOAT32` | float |
| `ACE_DTYPE_FLOAT64` | double |
| `ACE_DTYPE_INT32` | int |
| `ACE_DTYPE_INT64` | long |

### 设备类型

| 类型 | 描述 |
|------|------|
| `ACE_DEVICE_CPU` | CPU |
| `ACE_DEVICE_CUDA` | NVIDIA GPU |
| `ACE_DEVICE_OPENCL` | OpenCL 设备 |
| `ACE_DEVICE_VULKAN` | Vulkan 设备 |

## 架构说明

```
┌─────────────────────────────────────────┐
│           用户应用层                     │
│                                          │
│  ACE_KERNEL(my_kernel, { ... })         │
│  ace_kernel_invoke(...)                  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           核心引擎层                     │
│  - 设备管理  - 内存管理  - 内核调度      │
└─────────────────────────────────────────┘
                    ↓
┌─────────┬─────────┬──────────┬──────────┐
│  CPU    │  CUDA   │  OpenCL  │  Vulkan  │
│  JIT    │  NVRTC  │  Runtime │  SPIR-V  │
└─────────┴─────────┴──────────┴──────────┘
```

**设计原则：**
- Core 层只负责调度，不编译内核
- 后端层各自实现内核编译和执行
- 用户通过 ACE_KERNEL 宏定义自己的内核

## 许可证

MIT License
