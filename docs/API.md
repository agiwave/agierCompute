# AgierCompute API 参考

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
    /* 使用设备... */
    ace_device_release(dev);
}
```

---

### 内存管理

```c
ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);
void ace_buffer_free(ace_buffer_t buf);
ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);
ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);
```

---

### 内核管理

```c
/* 注册内核（通常使用 ACE_KERNEL 宏） */
ace_kernel_t ace_register_kernel(const char* name, const char* src);

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
int n = N;
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
ace_kernel_invoke(dev, kernel, ACE_DTYPE_FLOAT32, N, args, types, 4);
```

---

### 同步

```c
ace_error_t ace_finish(ace_device_t dev);  /* 等待完成 */
```

---

## 宏

### ACE_KERNEL

定义泛型内核。

```c
ACE_KERNEL(name, code)
```

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

错误检查。

```c
ACE_CHECK(ace_buffer_alloc(dev, size, &buf));
```

### ACE_INVOKE_1D

简化的内核调用（待实现）。

---

## 数据类型

```c
typedef enum {
    ACE_DTYPE_FLOAT32 = 0,
    ACE_DTYPE_FLOAT64 = 1,
    ACE_DTYPE_INT32   = 2,
    ACE_DTYPE_INT64   = 3,
} ace_dtype_t;
```

---

## 错误码

```c
#define ACE_OK              0
#define ACE_ERROR          -1
#define ACE_ERROR_MEM      -2
#define ACE_ERROR_DEVICE   -3
#define ACE_ERROR_COMPILE  -4
#define ACE_ERROR_LAUNCH   -5
```

---

## 测试框架

位于 `examples/lib/ace_test.h`。

```c
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
    /* 测试实现 */
    return ACE_TEST_PASS;
}

ace_test_case_t tests[] = {
    ACE_TEST_DEFINE("vec_add", test_vec_add, NULL),
};

ace_test_suite_t suite = { .name = "Tests", .tests = tests, .test_count = 1 };
ace_test_suite_run(&suite);
```
