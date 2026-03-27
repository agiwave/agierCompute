# ACE_INVOKE 宏使用指南

## 概述

`ACE_INVOKE` 是 AgierCompute 提供的简化内核调用宏，它自动识别参数类型，让用户无需手动构建参数数组。

## 基本用法

```c
#include "ace.h"

/* 1. 定义内核 */
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

/* 2. 执行内核 */
const int N = 1024;
float a[N], b[N], c[N];
ace_buffer_t ba, bb, bc;

/* 初始化 buffer ... */

/* 使用 ACE_INVOKE 宏 */
int n = N;
ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, n, ba, bb, bc);
```

## 参数类型自动识别

`ACE_INVOKE` 宏使用 C11 `_Generic` 自动识别参数类型：

| 参数类型 | 识别为 | 传递方式 |
|----------|--------|----------|
| `ace_buffer_t` | 缓冲区 | 直接传递指针 |
| `int`, `float`, `double` 等 | 标量值 | 传递地址 |

```c
/* 示例：混合参数 */
int n = 1024;
float alpha = 2.0f;
ace_buffer_t buf_a, buf_c;

/* 宏自动识别：n(标量), alpha(标量), buf_a(buffer), buf_c(buffer) */
ACE_INVOKE(dev, vec_scale, ACE_DTYPE_FLOAT32, N, n, alpha, buf_a, buf_c);
```

## ⚠️ 重要使用限制

### 标量参数必须是左值（变量）

`ACE_INVOKE` 宏需要对参数取地址，因此标量参数必须是**左值（有内存地址的变量）**。

#### ✅ 正确用法

```c
/* 使用变量 */
int n = 100;
float alpha = 2.0f;
ACE_INVOKE(dev, kernel, dtype, N, n, alpha, buf);

/* 使用已存在的变量 */
const int N = 1024;
int n_val = N;
ACE_INVOKE(dev, kernel, dtype, N, n_val, buf);
```

#### ❌ 错误用法

```c
/* 字面量 - 无法取地址 */
ACE_INVOKE(dev, kernel, dtype, N, 100, 2.0f, buf);  // 编译错误！

/* 表达式 - 无法取地址 */
ACE_INVOKE(dev, kernel, dtype, N, N + 1, buf);  // 编译错误！

/* 不必要的取地址 */
int n = 100;
ACE_INVOKE(dev, kernel, dtype, N, &n, buf);  // 错误！应传 n 而非 &n
```

### 常见场景及解决方案

| 场景 | 错误写法 | 正确写法 |
|------|----------|----------|
| 传递常量 | `ACE_INVOKE(..., 100, ...)` | `int n = 100; ACE_INVOKE(..., n, ...)` |
| 传递表达式 | `ACE_INVOKE(..., N*2, ...)` | `int n = N*2; ACE_INVOKE(..., n, ...)` |
| 传递计算结果 | `ACE_INVOKE(..., a+b, ...)` | `float sum = a+b; ACE_INVOKE(..., sum, ...)` |
| 传递数组大小 | `ACE_INVOKE(..., sizeof(arr), ...)` | `size_t sz = sizeof(arr); ACE_INVOKE(..., sz, ...)` |

## 完整示例

### 向量加法

```c
#include "ace.h"

/* 定义内核 */
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

int main() {
    /* 初始化 */
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CUDA, 0, &dev);
    
    const int N = 1024;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    /* 分配设备内存 */
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(float), &ba);
    ace_buffer_alloc(dev, N * sizeof(float), &bb);
    ace_buffer_alloc(dev, N * sizeof(float), &bc);
    
    ace_buffer_write(ba, h_a, N * sizeof(float));
    ace_buffer_write(bb, h_b, N * sizeof(float));
    
    /* 执行内核 - 注意使用变量 */
    int n = N;
    ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, n, ba, bb, bc);
    ace_finish(dev);
    
    /* 读取结果 */
    ace_buffer_read(bc, h_c, N * sizeof(float));
    
    /* 清理 */
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    ace_device_release(dev);
    
    return 0;
}
```

### 带标量参数的内核

```c
/* 定义带标量参数的内核 */
ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = alpha * a[i];
    }
);

/* 执行 */
const int N = 1024;
float alpha = 2.5f;
ace_buffer_t buf_a, buf_c;

/* 正确：使用变量 */
int n = N;
float alpha_val = alpha;
ACE_INVOKE(dev, vec_scale, ACE_DTYPE_FLOAT32, N, n, alpha_val, buf_a, buf_c);
```

## 技术细节

### 宏实现原理

`ACE_INVOKE` 宏内部：

1. 使用 `_Generic` 识别每个参数的类型
2. 对 `ace_buffer_t` 类型：直接使用指针值
3. 对其他类型：取地址后传递
4. 构建参数指针数组和大小数组
5. 调用 `ace_kernel_invoke`

```c
/* 简化版实现示意 */
#define _ACE_AP(a) _Generic((a), \
    ace_buffer_t: (void*)(a), \
    default: (void*)&(a) \
)

#define ACE_INVOKE(dev, kernel, dtype, n, ...) \
    ({ \
        void* _args[] = { _ACE_AP(__VA_ARGS__) }; \
        int _sizes[] = { /* 自动计算大小 */ }; \
        ace_kernel_invoke(dev, _ace_get_##kernel(), dtype, n, _args, _sizes, nargs); \
    })
```

### 参数数量限制

- **最大支持**: 8 个参数
- **原因**: 宏展开需要为每个参数生成代码

如需传递更多参数，请使用原始 API：

```c
/* 超过 8 个参数时使用原始 API */
void* args[] = {&n, &alpha, &beta, buf1, buf2, buf3, buf4, buf5, buf6};
int sizes[] = {sizeof(int), sizeof(float), sizeof(float), 0, 0, 0, 0, 0, 0};
ace_kernel_invoke(dev, _ace_get_kernel(), dtype, N, args, sizes, 9);
```

## 常见问题

### Q: 为什么不能直接传递字面量？

A: C 语言中，字面量（如 `100`, `3.14`）没有内存地址，无法对其使用 `&` 运算符。`ACE_INVOKE` 需要对参数取地址以传递给内核。

### Q: `const` 变量可以作为参数吗？

A: 可以。`const` 变量仍然是左值，有内存地址：

```c
const int N = 1024;
int n = N;  // 或者直接 const int n = N;
ACE_INVOKE(dev, kernel, dtype, N, n, buf);
```

### Q: 如何传递结构体？

A: 结构体作为标量处理（传递地址）：

```c
typedef struct { int x, y; } Point;
Point p = {10, 20};
ACE_INVOKE(dev, kernel, dtype, N, p, buf);
```

### Q: 如何传递数组指针？

A: 数组退化为指针，作为标量处理：

```c
int arr[10];
int* p = arr;
ACE_INVOKE(dev, kernel, dtype, N, p, buf);
```

## 相关文档

- [API 参考](API.md) - 完整 API 文档
- [README](../README.md) - 项目概述
