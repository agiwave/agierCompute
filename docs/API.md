# AgierCompute API 文档

## 概述

AgierCompute (ACE) 是一个跨平台 GPU 计算框架，提供统一的 API 支持 CPU、CUDA、OpenCL 和 Vulkan 后端。

## 快速开始

### 1. 包含头文件

```c
#include "ace.h"
```

### 2. 定义内核

```c
// 使用 ACE_KERNEL 宏定义泛型内核
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### 3. 基本使用流程

```c
// 获取设备
ace_device_t dev;
ace_device_get(ACE_DEVICE_CPU, 0, &dev);

// 分配内存
ace_buffer_t buf_a, buf_b, buf_c;
ace_buffer_alloc(dev, size, &buf_a);
// ...

// 写入数据
ace_buffer_write(buf_a, host_data, size);

// 执行内核
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);

// 等待完成
ace_finish(dev);

// 读取结果
ace_buffer_read(buf_c, host_result, size);

// 清理
ace_buffer_free(buf_a);
ace_device_release(dev);
```

---

## API 参考

### 设备管理

#### ace_device_count

获取指定类型设备的数量。

```c
ace_error_t ace_device_count(ace_device_type_t type, int* count);
```

**参数：**
- `type`: 设备类型 (ACE_DEVICE_CPU, ACE_DEVICE_CUDA, 等)
- `count`: 输出参数，设备数量

**返回值：**
- `ACE_OK`: 成功
- 其他：错误码

**示例：**
```c
int count;
ace_device_count(ACE_DEVICE_CPU, &count);
printf("CPU devices: %d\n", count);
```

---

#### ace_device_get

获取指定类型的设备句柄。

```c
ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev);
```

**参数：**
- `type`: 设备类型
- `idx`: 设备索引（从 0 开始）
- `dev`: 输出参数，设备句柄

**返回值：**
- `ACE_OK`: 成功
- `ACE_ERROR_DEVICE`: 设备不存在

---

#### ace_device_get_all

获取所有可用设备。

```c
ace_error_t ace_device_get_all(ace_device_list_t* list);
```

**参数：**
- `list`: 输出参数，设备列表

**返回值：**
- `ACE_OK`: 成功

**示例：**
```c
ace_device_list_t devices;
ace_device_get_all(&devices);
printf("Found %d devices\n", devices.count);
for (int i = 0; i < devices.count; i++) {
    ace_device_print_info(devices.devices[i]);
}
ace_device_list_release(&devices);
```

---

#### ace_device_select_best

自动选择最佳设备（优先 GPU）。

```c
ace_error_t ace_device_select_best(ace_device_t* dev);
```

**参数：**
- `dev`: 输出参数，最佳设备句柄

**返回值：**
- `ACE_OK`: 成功
- `ACE_ERROR_NOT_FOUND`: 无可用设备

---

#### ace_device_release

释放设备句柄。

```c
void ace_device_release(ace_device_t dev);
```

---

#### ace_device_props

获取设备属性。

```c
ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props);
```

**参数：**
- `dev`: 设备句柄
- `props`: 输出参数，设备属性结构体

**ace_device_props_t 结构：**
```c
typedef struct {
    ace_device_type_t type;     // 设备类型
    char name[256];             // 设备名称
    char vendor[128];           // 厂商
    size_t total_memory;        // 总内存（字节）
    size_t max_threads;         // 最大线程数
    int compute_units;          // 计算单元数
} ace_device_props_t;
```

---

#### ace_device_print_info

打印设备信息（辅助函数）。

```c
void ace_device_print_info(ace_device_t dev);
```

---

### 内存管理

#### ace_buffer_alloc

分配设备内存。

```c
ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);
```

**参数：**
- `dev`: 设备句柄
- `size`: 内存大小（字节）
- `buf`: 输出参数，缓冲区句柄

---

#### ace_buffer_free

释放设备内存。

```c
void ace_buffer_free(ace_buffer_t buf);
```

---

#### ace_buffer_write

写入数据到设备缓冲区（异步）。

```c
ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);
```

**参数：**
- `buf`: 缓冲区句柄
- `data`: 主机数据指针
- `size`: 数据大小（字节）

---

#### ace_buffer_read

从设备缓冲区读取数据（自动同步）。

```c
ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);
```

**参数：**
- `buf`: 缓冲区句柄
- `data`: 主机数据指针（输出）
- `size`: 数据大小（字节）

---

### 多设备内存管理

#### ace_buffer_alloc_sharded

在多个设备上分配分片缓冲区。

```c
ace_error_t ace_buffer_alloc_sharded(
    ace_device_list_t* devices,
    size_t total_size,
    ace_sharded_buffer_t* sharded
);
```

**参数：**
- `devices`: 设备列表
- `total_size`: 总大小（自动分片）
- `sharded`: 输出参数，分片缓冲区

**ace_sharded_buffer_t 结构：**
```c
typedef struct {
    ace_buffer_t* buffers;   // 各设备上的缓冲区
    size_t* offsets;         // 各分片偏移
    size_t* sizes;           // 各分片大小
    int count;               // 分片数量
} ace_sharded_buffer_t;
```

---

#### ace_buffer_free_sharded

释放分片缓冲区。

```c
void ace_buffer_free_sharded(ace_sharded_buffer_t* sharded);
```

---

#### ace_buffer_write_sharded

写入数据到分片缓冲区。

```c
ace_error_t ace_buffer_write_sharded(
    ace_sharded_buffer_t* sharded,
    const void* data,
    size_t total_size
);
```

---

#### ace_buffer_read_sharded

从分片缓冲区读取数据。

```c
ace_error_t ace_buffer_read_sharded(
    ace_sharded_buffer_t* sharded,
    void* data,
    size_t total_size
);
```

---

### 内核执行

#### ace_kernel_invoke

在单个设备上执行内核（1D 调度）。

```c
ace_error_t ace_kernel_invoke(
    ace_device_t dev,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
);
```

**参数：**
- `dev`: 设备句柄
- `kernel`: 内核句柄（使用 `_ace_get_xxx()` 获取）
- `dtype`: 数据类型（ACE_DTYPE_FLOAT32 等）
- `n`: 并行元素数量
- `args`: 参数数组
- `types`: 参数类型数组（ACE_VAL 或 ACE_BUF）
- `nargs`: 参数数量

**示例：**
```c
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
```

---

#### ace_kernel_launch

在单个设备上执行内核（自定义 3D 调度）。

```c
ace_error_t ace_kernel_launch(
    ace_device_t dev,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    ace_launch_config_t* config,
    void** args,
    int* types,
    int nargs
);
```

**ace_launch_config_t 结构：**
```c
typedef struct {
    size_t grid[3];      // 工作组数量
    size_t block[3];     // 每工作组线程数
    size_t shared_mem;   // 共享内存大小
} ace_launch_config_t;
```

**辅助函数：**
```c
ace_launch_config_t ace_launch_1d(size_t n, size_t block);
ace_launch_config_t ace_launch_2d(size_t nx, size_t ny, size_t bx, size_t by);
ace_launch_config_t ace_launch_3d(size_t nx, size_t ny, size_t nz,
                                   size_t bx, size_t by, size_t bz);
```

---

#### ace_kernel_invoke_sharded

在多个设备上并行执行内核。

```c
ace_error_t ace_kernel_invoke_sharded(
    ace_device_list_t* devices,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
);
```

---

### 同步

#### ace_finish

等待设备完成所有操作。

```c
ace_error_t ace_finish(ace_device_t dev);
```

---

#### ace_finish_all

等待所有设备完成所有操作。

```c
ace_error_t ace_finish_all(ace_device_list_t* devices);
```

---

### 内核注册

#### ace_register_kernel

注册内核。

```c
ace_kernel_t ace_register_kernel(const char* name, const char* src);
```

**注意：** 通常使用 `ACE_KERNEL` 宏自动注册。

---

## 内核语言

### 内建变量

在内核代码中可使用以下内建变量：

| 变量 | 描述 |
|------|------|
| `GID` | 全局线程 ID |
| `LID` | 局部线程 ID（工作组内） |
| `BSIZE` | 工作组大小 |
| `BARRIER()` | 工作组内屏障同步 |

### 类型占位符

使用 `T` 作为类型占位符，框架会自动实例化为具体类型：

- `float` (ACE_DTYPE_FLOAT32)
- `double` (ACE_DTYPE_FLOAT64)
- `int` (ACE_DTYPE_INT32)
- `long` (ACE_DTYPE_INT64)

### 内核示例

```c
// 向量加法
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

// 带缩放的内核
ACE_KERNEL(scale_add,
    void scale_add(int n, T alpha, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * alpha + b[i];
    }
);

// 使用屏障同步
ACE_KERNEL(reduce,
    void reduce(int n, T* data) {
        __shared__ T shared[256];
        int i = GID;
        shared[LID] = data[i];
        BARRIER();
        // ... 归约操作
    }
);
```

---

## 错误处理

### 错误码

| 错误码 | 值 | 描述 |
|--------|-----|------|
| `ACE_OK` | 0 | 成功 |
| `ACE_ERROR` | -1 | 一般错误 |
| `ACE_ERROR_MEM` | -2 | 内存错误 |
| `ACE_ERROR_DEVICE` | -3 | 设备错误 |
| `ACE_ERROR_COMPILE` | -4 | 编译错误 |
| `ACE_ERROR_LAUNCH` | -5 | 启动错误 |
| `ACE_ERROR_IO` | -6 | I/O 错误 |
| `ACE_ERROR_BACKEND` | -7 | 后端错误 |
| `ACE_ERROR_NOT_FOUND` | -8 | 未找到 |
| `ACE_ERROR_INVALID` | -9 | 无效参数 |

### 错误处理示例

```c
ace_error_t err = ace_buffer_alloc(dev, size, &buf);
if (err != ACE_OK) {
    fprintf(stderr, "Buffer alloc failed: %s\n", ace_error_string(err));
    return err;
}
```

---

## 数据类型

| 枚举值 | C 类型 | 大小 |
|--------|--------|------|
| `ACE_DTYPE_FLOAT32` | `float` | 4 字节 |
| `ACE_DTYPE_FLOAT64` | `double` | 8 字节 |
| `ACE_DTYPE_INT32` | `int` | 4 字节 |
| `ACE_DTYPE_INT64` | `long` | 8 字节 |

---

## 参数类型

| 宏 | 值 | 描述 |
|-----|-----|------|
| `ACE_VAL` | 0 | 标量值（传指针） |
| `ACE_BUF` | 1 | 缓冲区（传 ace_buffer_t） |

---

## 最佳实践

### 1. 设备选择

```c
// 自动选择最佳设备
ace_device_t dev;
ace_device_select_best(&dev);

// 或指定使用 CPU
ace_device_get(ACE_DEVICE_CPU, 0, &dev);
```

### 2. 批量操作

```c
// 分配多个缓冲区
ace_buffer_t bufs[10];
for (int i = 0; i < 10; i++) {
    ace_buffer_alloc(dev, size, &bufs[i]);
}

// 批量释放
for (int i = 0; i < 10; i++) {
    ace_buffer_free(bufs[i]);
}
```

### 3. 多设备并行

```c
ace_device_list_t devices;
ace_device_get_all(&devices);

// 分配分片缓冲区
ace_sharded_buffer_t sharded;
ace_buffer_alloc_sharded(&devices, total_size, &sharded);

// 并行执行
ace_kernel_invoke_sharded(&devices, kernel, dtype, n, args, types, nargs);

// 等待所有设备
ace_finish_all(&devices);
```

### 4. 错误检查宏

```c
#define ACE_CHECK(call) do { \
    ace_error_t err = (call); \
    if (err != ACE_OK) { \
        fprintf(stderr, "ACE error at %s:%d: %s\n", \
                __FILE__, __LINE__, ace_error_string(err)); \
        return err; \
    } \
} while(0)

// 使用
ACE_CHECK(ace_buffer_alloc(dev, size, &buf));
```

---

## 版本信息

```c
#define ACE_VERSION_MAJOR 1
#define ACE_VERSION_MINOR 0
#define ACE_VERSION_PATCH 0
#define ACE_VERSION "1.0.0"
```

获取版本字符串：
```c
printf("Using AgierCompute %s\n", ACE_VERSION);
```
