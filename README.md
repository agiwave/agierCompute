# AgierCompute - 跨平台 GPU 计算框架

一个类似 SYCL 风格的跨平台并行计算框架，支持 CPU、CUDA、OpenCL 和 Vulkan 后端。

## 特性

- **统一的 API 设计**：简洁的 C 语言 API，类似 SYCL 风格
- **多后端支持**：
  - CPU（多线程并行）
  - CUDA（NVIDIA GPU，支持 NVRTC 运行时编译）
  - OpenCL（跨平台 GPU/CPU）
  - Vulkan（现代图形 API 计算）
- **跨 GPU 运行**：自动发现设备，支持数据并行分片执行
- **内核模板**：泛型内核定义，自动类型实例化
- **内建变量支持**：`GID`（全局 ID）、`LID`（局部 ID）、`BSIZE`（工作组大小）、`BARRIER()`（屏障）

## 快速开始

### 1. 定义内核

```c
#include "ace.h"

// 定义向量加法内核（T 是类型占位符）
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### 2. 获取设备

```c
// 获取所有可用设备
ace_device_list_t devices;
ace_device_get_all(&devices);

// 或选择最佳设备（优先 GPU）
ace_device_t dev;
ace_device_select_best(&dev);
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
float h_a[N], h_b[N];
// ... 初始化数据 ...
ace_buffer_write(buf_a, h_a, N * sizeof(float));
ace_buffer_write(buf_b, h_b, N * sizeof(float));
```

### 5. 执行内核

```c
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

// 单设备执行
ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);

// 或多设备数据并行
ace_kernel_invoke_sharded(&devices, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N,
                          args, types, 4);
```

### 6. 读取结果

```c
float h_c[N];
ace_buffer_read(buf_c, h_c, N * sizeof(float));
```

### 7. 清理

```c
ace_buffer_free(buf_a);
ace_buffer_free(buf_b);
ace_buffer_free(buf_c);
ace_device_release(dev);
ace_device_list_release(&devices);
```

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行测试

```bash
# 简单测试（CPU 后端）
./bin/simple_test

# 多后端演示
./bin/ace_demo

# 多设备测试
./bin/multi_device_test
```

## API 参考

### 设备管理

| 函数 | 描述 |
|------|------|
| `ace_device_get_all()` | 获取所有可用设备 |
| `ace_device_select_best()` | 选择最佳设备（优先 GPU） |
| `ace_device_get()` | 获取指定类型的设备 |
| `ace_device_count()` | 获取设备数量 |
| `ace_device_release()` | 释放设备 |
| `ace_device_props()` | 获取设备属性 |
| `ace_device_print_info()` | 打印设备信息 |

### 内存管理

| 函数 | 描述 |
|------|------|
| `ace_buffer_alloc()` | 分配设备内存 |
| `ace_buffer_free()` | 释放设备内存 |
| `ace_buffer_write()` | 写入数据到设备 |
| `ace_buffer_read()` | 从设备读取数据 |
| `ace_buffer_alloc_sharded()` | 在多设备上分配分片缓冲区 |
| `ace_buffer_free_sharded()` | 释放分片缓冲区 |
| `ace_buffer_write_sharded()` | 写入分片缓冲区 |
| `ace_buffer_read_sharded()` | 读取分片缓冲区 |

### 内核执行

| 函数 | 描述 |
|------|------|
| `ace_kernel_invoke()` | 单设备执行内核（1D 调度） |
| `ace_kernel_launch()` | 执行内核（自定义 3D 调度） |
| `ace_kernel_invoke_sharded()` | 多设备数据并行执行 |
| `ace_register_kernel()` | 注册内核 |

### 同步

| 函数 | 描述 |
|------|------|
| `ace_finish()` | 等待设备完成所有操作 |
| `ace_finish_all()` | 等待所有设备完成 |

## 支持的内核

框架预置支持以下内核操作：

- **向量运算**：`vec_add`, `vec_sub`, `vec_mul`
- **激活函数**：`relu`, `sigmoid`, `tanh`, `softmax`
- **数学函数**：`exp`, `log`, `sqrt`, `square`, `abs`
- **线性代数**：`gemm`（矩阵乘法）, `dot`（点积）
- **数据操作**：`scale`（缩放）, `fill`（填充）, `copy`（拷贝）, `negate`（取反）

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    用户应用层                            │
│                   (ace.h API)                           │
├─────────────────────────────────────────────────────────┤
│                    核心引擎层                            │
│  - 设备管理  - 内存管理  - 内核编译  - 调度执行          │
│                   (engine.c)                            │
├───────────┬───────────┬───────────┬───────────┬────────┤
│  CPU 后端  │ CUDA 后端  │ OpenCL 后端 │ Vulkan 后端 │ ...   │
│ (多线程)  │ (NVRTC)   │ (标准)    │ (SPIR-V)  │       │
└───────────┴───────────┴───────────┴───────────┴────────┘
```

## 内建变量

在内核代码中可以使用以下内建变量：

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID（工作组内） |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 工作组内屏障同步 |

## 数据类型

| 类型 | 枚举值 | C 类型 |
|------|--------|--------|
| `ACE_DTYPE_FLOAT32` | `float` | 32 位浮点数 |
| `ACE_DTYPE_FLOAT64` | `double` | 64 位浮点数 |
| `ACE_DTYPE_INT32` | `int` | 32 位整数 |
| `ACE_DTYPE_INT64` | `long` | 64 位整数 |

## 错误码

| 错误码 | 值 | 描述 |
|--------|-----|------|
| `ACE_OK` | 0 | 成功 |
| `ACE_ERROR` | -1 | 一般错误 |
| `ACE_ERROR_MEM` | -2 | 内存错误 |
| `ACE_ERROR_DEVICE` | -3 | 设备错误 |
| `ACE_ERROR_COMPILE` | -4 | 编译错误 |
| `ACE_ERROR_LAUNCH` | -5 | 启动错误 |
| `ACE_ERROR_BACKEND` | -7 | 后端错误 |

## 示例项目结构

```
agierCompute/
├── include/
│   └── ace.h              # 公共 API 头文件
├── src/
│   ├── core/
│   │   ├── engine.c       # 核心引擎
│   │   └── backend_loader.c
│   ├── cpu/
│   │   └── cpu_backend.c  # CPU 后端
│   ├── cuda/
│   │   └── cuda_backend.c # CUDA 后端
│   ├── opencl/
│   │   └── opencl_backend.c
│   └── vulkan/
│       └── vulkan_backend.c
├── examples/
│   ├── demo.cpp           # 多后端演示
│   ├── simple_test.c      # 简单测试
│   ├── user_kernels.c     # 用户自定义内核
│   └── multi_device_test.c # 多设备测试
└── CMakeLists.txt
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
