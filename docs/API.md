# AgierCompute API 参考

## 概述

AgierCompute 是一个跨平台 GPU 计算框架，采用表驱动架构设计，让用户轻松编写跨 GPU 内核。

### 核心设计理念

**不提供预置内核**，只提供让用户方便编写自己内核的工具。

### 架构特点

- **表驱动设计**: 所有后端采用统一的数据类型表和内核操作表
- **高内聚**: 每个操作的完整实现在一个 inject 函数内
- **低耦合**: 类型和操作完全隔离
- **易扩展**: 添加新类型只需在类型表中添加条目

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

---

## 核心 API

### 设备管理

```c
/* 获取设备数量（type=ACE_DEVICE_ALL 遍历所有后端） */
ace_error_t ace_device_count(ace_device_type_t type, int* count);

/* 获取设备 */
ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev);

/* 释放设备 */
void ace_device_release(ace_device_t dev);

/* 获取设备属性 */
ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props);
```

**设备类型：**
```c
typedef enum {
    ACE_DEVICE_CPU    = 0,   /* 未实现 */
    ACE_DEVICE_CUDA   = 1,
    ACE_DEVICE_OPENCL = 2,
    ACE_DEVICE_VULKAN = 3,
    ACE_DEVICE_METAL  = 4,   /* 未实现 */
    ACE_DEVICE_ALL    = 5,   /* 遍历所有可用设备 */
} ace_device_type_t;
```

**示例：**
```c
/* 遍历所有设备 */
int count;
ace_device_count(ACE_DEVICE_ALL, &count);
for (int i = 0; i < count; i++) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_ALL, i, &dev);
    
    ace_device_props_t props;
    ace_device_props(dev, &props);
    printf("Device: %s\n", props.name);
    
    ace_device_release(dev);
}
```

---

### 内存管理

```c
/* 分配设备内存 */
ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);

/* 释放设备内存 */
void ace_buffer_free(ace_buffer_t buf);

/* 写入数据到设备（异步） */
ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);

/* 从设备读取数据（自动同步） */
ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);
```

**示例：**
```c
ace_buffer_t buf;
ace_buffer_alloc(dev, 1024 * sizeof(float), &buf);

float data[1024];
// ... 初始化数据 ...

ace_buffer_write(buf, data, sizeof(data));  // 异步写入
// ... 执行内核 ...
ace_buffer_read(buf, data, sizeof(data));   // 读取（自动同步）

ace_buffer_free(buf);
```

---

### 内核管理

```c
/* 注册内核（通常使用 ACE_KERNEL 宏） */
ace_kernel_t ace_kernel_register(const char* name, const char* src);

/* 执行内核 */
ace_error_t ace_kernel_invoke(
    ace_device_t dev, ace_kernel_t kernel, ace_dtype_t dtype, size_t n,
    void** args, int* types, int nargs);

/* 执行内核（自定义 3D 调度） */
ace_error_t ace_kernel_launch(
    ace_device_t dev, ace_kernel_t kernel, ace_dtype_t dtype,
    ace_launch_config_t* config, void** args, int* types, int nargs);
```

**参数类型：**
- `ACE_VAL` (0) - 标量值
- `ACE_BUF` (1) - 缓冲区

**示例：**
```c
// 2 个标量 + 2 个缓冲区
int n = N;
float alpha = 2.0f;
void* args[] = {&n, &alpha, buf_a, buf_c};
int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};

ace_kernel_invoke(dev, kernel, ACE_DTYPE_FLOAT32, N, args, types, 4);
```

---

### 同步

```c
/* 等待设备上所有操作完成 */
ace_error_t ace_finish(ace_device_t dev);
```

**注意**: `ace_buffer_read` 会自动同步，通常不需要显式调用 `ace_finish`。

---

## 宏

### ACE_KERNEL

定义泛型内核。

```c
ACE_KERNEL(name, code)
```

**内核内建变量：**
- `GID` - 全局线程 ID
- `LID` - 局部线程 ID  
- `BSIZE` - 工作组大小
- `BARRIER()` - 屏障同步

**示例：**
```c
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);
```

### ACE_CHECK

错误检查（用于返回 ace_error_t 的函数）。

```c
ACE_CHECK(ace_buffer_alloc(dev, size, &buf));
```

### ACE_CHECK_VOID

错误检查（用于 void 函数）。

```c
ACE_CHECK_VOID(some_void_function());
```

### ACE_INVOKE

简化的内核调用宏（C 语言）。

```c
ACE_INVOKE(dev, kernel_name, dtype, n, arg1, arg2, ...);
```

**参数说明：**
- `dev` - 设备句柄
- `kernel_name` - 内核名称（与 `ACE_KERNEL` 定义的名称一致）
- `dtype` - 数据类型（`ace_dtype_t`）
- `n` - 全局线程数量
- `arg1, arg2, ...` - 内核参数（支持最多 8 个参数）

**参数类型自动识别：**
- `ace_buffer_t` 类型 → 自动识别为缓冲区（传递指针）
- 其他类型 → 自动识别为标量值（传递地址）

**⚠️ 重要使用限制：**

标量参数必须是**左值（变量）**，不能是字面量或表达式。

| 用法 | 示例 | 状态 |
|------|------|------|
| 变量 | `int n = 100; ACE_INVOKE(..., n, ...);` | ✅ 正确 |
| 字面量 | `ACE_INVOKE(..., 100, ...);` | ❌ 编译错误 |
| 表达式 | `ACE_INVOKE(..., N + 1, ...);` | ❌ 编译错误 |
| 取地址 | `ACE_INVOKE(..., &n, ...);` | ❌ 错误（应传 `n`） |

**正确用法示例：**

```c
/* 定义内核 */
ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = alpha * a[i];
    }
);

/* 执行内核 */
const int N = 1024;
float alpha = 2.0f;
ace_buffer_t buf_a, buf_c;

/* 正确：使用变量 */
int n = N;
ACE_INVOKE(dev, vec_scale, ACE_DTYPE_FLOAT32, N, n, alpha, buf_a, buf_c);

/* 错误：字面量会导致编译错误 */
// ACE_INVOKE(dev, vec_scale, ACE_DTYPE_FLOAT32, N, N, 2.0f, buf_a, buf_c);  // ❌

/* 解决方法：先赋值给变量 */
int n_val = N;
float alpha_val = 2.0f;
ACE_INVOKE(dev, vec_scale, ACE_DTYPE_FLOAT32, N, n_val, alpha_val, buf_a, buf_c);  // ✅
```

**技术说明：**

`ACE_INVOKE` 宏使用 C11 `_Generic` 实现类型自动识别：
- 内部维护参数指针数组和大小数组
- 自动区分 buffer（传递指针）和标量（传递地址）
- 支持 1-8 个参数

如需传递更多参数或需要更灵活的控制，请使用原始 `ace_kernel_invoke` API。

---

## 数据类型

```c
typedef enum {
    ACE_DTYPE_FLOAT32 = 0,   /* 单精度浮点 */
    ACE_DTYPE_FLOAT64 = 1,   /* 双精度浮点 */
    ACE_DTYPE_INT32   = 2,   /* 32 位整数 */
    ACE_DTYPE_INT64   = 3,   /* 64 位整数 */
    ACE_DTYPE_FLOAT16 = 4,   /* 半精度浮点 (AI 推理) */
    ACE_DTYPE_BFLOAT16 = 5,  /* Brain 浮点 (AI 训练) */
    ACE_DTYPE_INT8    = 6,   /* 8 位整数 (量化) */
    ACE_DTYPE_UINT8   = 7,   /* 8 位无符号整数 */
    ACE_DTYPE_INT16   = 8,   /* 16 位整数 */
    ACE_DTYPE_BOOL    = 9,   /* 布尔值 */
} ace_dtype_t;
```

### 类型转换辅助函数

```c
/* FP16 转换 */
uint16_t float_to_float16(float f);
float float16_to_float(uint16_t h);

/* BF16 转换 */
uint16_t float_to_bfloat16(float f);
float bfloat16_to_float(uint16_t h);
```

**示例：**
```c
// FLOAT16 标量参数
uint16_t alpha = float_to_float16(2.0f);
ace_kernel_invoke(dev, kernel, ACE_DTYPE_FLOAT16, N,
                  (void*[]){&n, &alpha, buf_a, buf_c},
                  (int[]){sizeof(int), sizeof(uint16_t), 0, 0}, 4);
```

---

## 错误码

```c
#define ACE_OK              0   /* 成功 */
#define ACE_ERROR          -1   /* 一般错误 */
#define ACE_ERROR_MEM      -2   /* 内存错误 */
#define ACE_ERROR_DEVICE   -3   /* 设备错误 */
#define ACE_ERROR_COMPILE  -4   /* 编译错误 */
#define ACE_ERROR_LAUNCH   -5   /* 启动错误 */
#define ACE_ERROR_IO       -6   /* I/O 错误 */
#define ACE_ERROR_BACKEND  -7   /* 后端错误 */
#define ACE_ERROR_NOT_FOUND -8  /* 未找到 */
#define ACE_ERROR_INVALID   -9  /* 无效参数 */
```

**错误处理：**
```c
const char* ace_error_string(ace_error_t err);  /* 获取错误描述 */
```

---

## 3D 调度配置

```c
typedef struct ace_launch_config_ {
    size_t grid[3];     /* 工作组数量 */
    size_t block[3];    /* 每个工作组的线程数 */
    size_t shared_mem;  /* 动态共享内存大小 */
} ace_launch_config_t;

/* 辅助函数 */
ace_launch_config_t ace_launch_1d(size_t n, size_t block);
ace_launch_config_t ace_launch_2d(size_t nx, size_t ny, size_t bx, size_t by);
ace_launch_config_t ace_launch_3d(size_t nx, size_t ny, size_t nz,
                                   size_t bx, size_t by, size_t bz);

/* 简化宏 */
#define ACE_1D(n) ace_launch_1d(n, 256)
#define ACE_1D_BLOCK(n, b) ace_launch_1d(n, b)
```

---

## 测试框架

位于 `examples/lib/ace_test.h`。

```c
/* 测试结果 */
typedef enum {
    ACE_TEST_PASS,
    ACE_TEST_FAIL,
    ACE_TEST_SKIP
} ace_test_result_t;

/* 测试用例 */
typedef struct {
    const char* name;
    ace_test_result_t (*func)(ace_device_t, void*);
    void* user_data;
} ace_test_case_t;

/* 测试套件 */
typedef struct {
    const char* name;
    ace_test_case_t* tests;
    int test_count;
    int passed, failed, skipped;
} ace_test_suite_t;

/* 运行测试（自动遍历所有设备） */
void ace_test_suite_run(ace_test_suite_t* suite);
```

**示例：**
```c
static ace_test_result_t test_vec_add(ace_device_t dev, void* user_data) {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // ... 初始化数据、分配缓冲区、执行内核 ...
    
    // 验证结果
    for (int i = 0; i < N; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
            return ACE_TEST_FAIL;
        }
    }
    return ACE_TEST_PASS;
}

ace_test_case_t tests[] = {
    ACE_TEST_DEFINE("vec_add", test_vec_add, NULL),
};

ace_test_suite_t suite = { 
    .name = "Vector Tests", 
    .tests = tests, 
    .test_count = 1 
};
ace_test_suite_run(&suite);
```

---

## 后端架构

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

每个后端包含：
- **数据类型表** (`*_dtype_table.h/c`): 定义所有支持的数据类型信息
- **内核操作表** (`*_kernel_ops_table.h/c`): 定义所有内核操作

详见 [BACKEND_STATUS.md](BACKEND_STATUS.md)。

---

## 支持的后端

| 后端 | 状态 | 编译方式 |
|------|------|----------|
| CUDA | ✅ | NVRTC |
| OpenCL | ✅ | 运行时 |
| Vulkan | ✅ | shaderc |
| CPU | ❌ | - |
| Metal | ❌ | - |

> 使用 `ACE_DEVICE_ALL` 可遍历所有可用设备。
