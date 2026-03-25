# AgierCompute - 跨平台 GPU 计算框架

一个简洁的跨平台并行计算框架，支持 CPU、CUDA、OpenCL 和 Vulkan 后端。

## 快速开始

### 1. 定义内核

```c
#include "ace.h"

// 定义向量加法内核（T 是类型占位符，自动替换为 float/int/double）
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### 2. 获取设备

```c
// 获取 CPU 设备
ace_device_t dev;
ace_device_get(ACE_DEVICE_CPU, 0, &dev);

// 或获取 CUDA 设备
// ace_device_get(ACE_DEVICE_CUDA, 0, &dev);

// 或获取 Vulkan 设备
// ace_device_get(ACE_DEVICE_VULKAN, 0, &dev);
```

### 3. 分配内存

```c
ace_buffer_t buf_a, buf_b, buf_c;
ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
ace_buffer_alloc(dev, N * sizeof(float), &buf_c);
```

### 4. 写入数据

```c
float h_a[N] = {1, 2, 3, 4, 5};
float h_b[N] = {10, 20, 30, 40, 50};

ace_buffer_write(buf_a, h_a, N * sizeof(float));
ace_buffer_write(buf_b, h_b, N * sizeof(float));
```

### 5. 执行内核

```c
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

// 执行内核（自动编译、自动调度）
ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);

// 等待完成
ace_finish(dev);
```

### 6. 读取结果

```c
float h_c[N];
ace_buffer_read(buf_c, h_c, N * sizeof(float));
// h_c = {11, 22, 33, 44, 55}
```

### 7. 清理

```c
ace_buffer_free(buf_a);
ace_buffer_free(buf_b);
ace_buffer_free(buf_c);
ace_device_release(dev);
```

## 完整示例

```c
#include <stdio.h>
#include "ace.h"

// 定义内核
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

int main() {
    // 获取设备
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);
    
    // 准备数据
    const int N = 5;
    float h_a[] = {1, 2, 3, 4, 5};
    float h_b[] = {10, 20, 30, 40, 50};
    float h_c[N];
    
    // 分配设备内存
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);
    
    // 写入数据
    ace_buffer_write(buf_a, h_a, N * sizeof(float));
    ace_buffer_write(buf_b, h_b, N * sizeof(float));
    
    // 执行内核
    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    
    // 读取结果
    ace_buffer_read(buf_c, h_c, N * sizeof(float));
    
    // 打印结果
    for (int i = 0; i < N; i++) {
        printf("%g ", h_c[i]);
    }
    printf("\n");
    
    // 清理
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
    
    return 0;
}
```

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## 运行测试

```bash
# CPU 后端测试
./bin/simple_test

# CUDA 后端测试
./bin/cuda_test

# OpenCL 后端测试
./bin/opencl_test

# Vulkan 后端测试（支持所有设备：NVIDIA/Intel/llvmpipe）
./bin/vulkan_test
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
| `ace_buffer_write(buf, data, size)` | 写入数据（主机→设备） |
| `ace_buffer_read(buf, data, size)` | 读取数据（设备→主机） |
| `ace_kernel_invoke(dev, kernel, dtype, n, args, types, nargs)` | 执行内核 |
| `ace_finish(dev)` | 等待设备完成 |

### 数据类型

| 类型 | 描述 |
|------|------|
| `ACE_DTYPE_FLOAT32` | 32 位浮点数 (float) |
| `ACE_DTYPE_FLOAT64` | 64 位浮点数 (double) |
| `ACE_DTYPE_INT32` | 32 位整数 (int) |
| `ACE_DTYPE_INT64` | 64 位整数 (long) |

### 设备类型

| 类型 | 描述 |
|------|------|
| `ACE_DEVICE_CPU` | CPU（多线程并行） |
| `ACE_DEVICE_CUDA` | NVIDIA GPU（NVRTC 编译） |
| `ACE_DEVICE_OPENCL` | OpenCL 设备 |
| `ACE_DEVICE_VULKAN` | Vulkan 设备 |

### 内核内建变量

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID（工作组内） |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 工作组屏障同步 |

## 支持的后端

| 后端 | 状态 | 说明 |
|------|------|------|
| CPU | ✅ 完整支持 | 多线程并行，自动 JIT 编译 |
| CUDA | ✅ 完整支持 | NVRTC 运行时编译 |
| OpenCL | ✅ 完整支持 | 运行时编译 |
| Vulkan | ✅ 完整支持 | shaderc SPIR-V 编译 |

## 架构说明

```
┌─────────────────────────────────────────┐
│           用户应用层                     │
│         (使用 ace.h API)                 │
├─────────────────────────────────────────┤
│           核心引擎层                     │
│  - 设备管理  - 内存管理  - 内核调度      │
│  (engine.c - 只负责调度，不编译)          │
├─────────┬─────────┬──────────┬──────────┤
│  CPU    │  CUDA   │  OpenCL  │  Vulkan  │
│  后端   │  后端   │  后端    │  后端    │
│ (JIT)   │ (NVRTC) │ (Runtime)│(shaderc) │
└─────────┴─────────┴──────────┴──────────┘
```

- **Core 层**: 只负责设备管理、内存管理、内核调度
- **后端层**: 各自实现内核编译和缓存
  - CPU: 多线程执行
  - CUDA: NVRTC 运行时编译
  - OpenCL: 运行时编译
  - Vulkan: shaderc 编译 SPIR-V，每设备独立缓存

## 许可证

MIT License
