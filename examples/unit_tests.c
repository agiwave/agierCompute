/**
 * @file unit_tests.c
 * @brief AgierCompute 完整单元测试套件
 *
 * 测试覆盖：
 * - 设备管理 API
 * - 内存管理 API
 * - 所有预置内核
 * - 多设备并行
 * - 错误处理
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 测试宏和辅助函数
 * ============================================================================ */

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (at line %d)\n", msg, __LINE__); \
        return 0; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (got %f, expected %f, at line %d)\n", msg, (double)(a), (double)(b), __LINE__); \
        return 0; \
    } \
} while(0)

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(test_fn) do { \
    g_tests_run++; \
    printf("Running %s... ", #test_fn); \
    if (test_fn()) { \
        g_tests_passed++; \
        printf("PASS\n"); \
    } \
} while(0)

/* ============================================================================
 * 内核定义
 * ============================================================================ */

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

ACE_KERNEL(vec_sub,
    void vec_sub(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] - b[i];
    }
);

ACE_KERNEL(vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
);

ACE_KERNEL(scale,
    void scale(int n, T alpha, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
);

ACE_KERNEL(relu,
    void relu(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
);

ACE_KERNEL(sigmoid,
    void sigmoid(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
);

ACE_KERNEL(tanh_kernel,
    void tanh_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = tanhf(in[i]);
    }
);

ACE_KERNEL(abs_kernel,
    void abs_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] < 0 ? -in[i] : in[i];
    }
);

ACE_KERNEL(exp_kernel,
    void exp_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = expf(in[i]);
    }
);

ACE_KERNEL(log_kernel,
    void log_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = logf(in[i]);
    }
);

ACE_KERNEL(sqrt_kernel,
    void sqrt_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = sqrtf(in[i]);
    }
);

ACE_KERNEL(square,
    void square(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * in[i];
    }
);

ACE_KERNEL(fill,
    void fill(int n, T val, T* out) {
        int i = GID;
        if (i < n) out[i] = val;
    }
);

ACE_KERNEL(copy,
    void copy(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i];
    }
);

ACE_KERNEL(negate,
    void negate(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = -in[i];
    }
);

ACE_KERNEL(gemm,
    void gemm(int n, int m, int k, T* A, T* B, T* C) {
        int row = GID;
        if (row < n) {
            for (int j = 0; j < m; j++) {
                T sum = 0;
                for (int i = 0; i < k; i++) {
                    sum += A[row * k + i] * B[i * m + j];
                }
                C[row * m + j] = sum;
            }
        }
    }
);

/* ============================================================================
 * 设备管理测试
 * ============================================================================ */

static int test_device_count(void) {
    int count = 0;
    ace_error_t err;

    err = ace_device_count(ACE_DEVICE_CPU, &count);
    TEST_ASSERT(err == ACE_OK, "ace_device_count CPU");
    TEST_ASSERT(count >= 1, "CPU device available");

    /* 其他后端可能不可用，不强制要求 */
    ace_device_count(ACE_DEVICE_CUDA, &count);
    ace_device_count(ACE_DEVICE_OPENCL, &count);
    ace_device_count(ACE_DEVICE_VULKAN, &count);

    return 1;
}

static int test_device_get_all(void) {
    ace_device_list_t devices;
    ace_error_t err = ace_device_get_all(&devices);

    TEST_ASSERT(err == ACE_OK, "ace_device_get_all");
    TEST_ASSERT(devices.count >= 1, "At least one device");

    ace_device_list_release(&devices);
    return 1;
}

static int test_device_select_best(void) {
    ace_device_t dev;
    ace_error_t err = ace_device_select_best(&dev);

    TEST_ASSERT(err == ACE_OK, "ace_device_select_best");
    TEST_ASSERT(dev != NULL, "Best device not null");

    if (dev) ace_device_release(dev);
    return 1;
}

static int test_device_props(void) {
    ace_device_t dev;
    ace_error_t err = ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    TEST_ASSERT(err == ACE_OK, "Get CPU device");

    ace_device_props_t props;
    err = ace_device_props(dev, &props);
    TEST_ASSERT(err == ACE_OK, "Get device props");
    TEST_ASSERT(props.type == ACE_DEVICE_CPU, "Device type is CPU");
    TEST_ASSERT(props.max_threads > 0, "Max threads > 0");

    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内存管理测试
 * ============================================================================ */

static int test_buffer_alloc_free(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    ace_buffer_t buf;
    ace_error_t err = ace_buffer_alloc(dev, 1024, &buf);
    TEST_ASSERT(err == ACE_OK, "Buffer alloc");
    TEST_ASSERT(buf != NULL, "Buffer not null");

    ace_buffer_free(buf);
    ace_device_release(dev);
    return 1;
}

static int test_buffer_write_read(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 100;
    float h_data[N], h_result[N];
    for (int i = 0; i < N; i++) h_data[i] = i * 1.5f;

    ace_buffer_t buf;
    ace_buffer_alloc(dev, N * sizeof(float), &buf);

    ace_error_t err = ace_buffer_write(buf, h_data, N * sizeof(float));
    TEST_ASSERT(err == ACE_OK, "Buffer write");

    err = ace_buffer_read(buf, h_result, N * sizeof(float));
    TEST_ASSERT(err == ACE_OK, "Buffer read");

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(h_data[i], h_result[i], 1e-6f, "Data match");
    }

    ace_buffer_free(buf);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内核测试 - 向量运算
 * ============================================================================ */

static int test_vec_add_float(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 1000;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

    ace_buffer_write(buf_a, a, N * sizeof(float));
    ace_buffer_write(buf_b, b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    ace_error_t err = ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    TEST_ASSERT(err == ACE_OK, "Kernel invoke");
    ace_finish(dev);

    ace_buffer_read(buf_c, c, N * sizeof(float));

    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_NEAR(c[i], a[i] + b[i], 1e-5f, "vec_add result");
    }

    free(a); free(b); free(c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
    return 1;
}

static int test_vec_sub_float(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 100;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 3.0f;
        b[i] = i * 1.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

    ace_buffer_write(buf_a, a, N * sizeof(float));
    ace_buffer_write(buf_b, b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_vec_sub(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(buf_c, c, N * sizeof(float));

    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_NEAR(c[i], a[i] - b[i], 1e-5f, "vec_sub result");
    }

    free(a); free(b); free(c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
    return 1;
}

static int test_vec_mul_float(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 100;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 0.5f;
        b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

    ace_buffer_write(buf_a, a, N * sizeof(float));
    ace_buffer_write(buf_b, b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_vec_mul(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(buf_c, c, N * sizeof(float));

    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_NEAR(c[i], a[i] * b[i], 1e-4f, "vec_mul result");
    }

    free(a); free(b); free(c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内核测试 - 激活函数
 * ============================================================================ */

static int test_relu(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);  /* 使用 CPU 后端更快 */

    const int N = 10;
    float input[] = {-5, -2, -1, 0, 1, 2, 3, 4, 5, 10};
    float expected[] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 10};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);

    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_relu(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], expected[i], 1e-5f, "ReLU result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_sigmoid(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 5;
    float input[] = {-2, -1, 0, 1, 2};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);

    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_sigmoid(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    /* sigmoid(0) = 0.5 */
    TEST_ASSERT_NEAR(output[2], 0.5f, 1e-5f, "sigmoid(0)");
    /* sigmoid(1) ≈ 0.731 */
    TEST_ASSERT_NEAR(output[3], 0.731f, 0.01f, "sigmoid(1)");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_tanh(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 5;
    float input[] = {-1, -0.5, 0, 0.5, 1};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);

    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_tanh_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    /* tanh(0) = 0 */
    TEST_ASSERT_NEAR(output[2], 0.0f, 1e-5f, "tanh(0)");
    /* tanh(1) ≈ 0.762 */
    TEST_ASSERT_NEAR(output[4], 0.762f, 0.01f, "tanh(1)");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内核测试 - 数学函数
 * ============================================================================ */

static int test_abs(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 6;
    float input[] = {-5, -2, 0, 2, 5, 10};
    float expected[] = {5, 2, 0, 2, 5, 10};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_abs_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], expected[i], 1e-5f, "abs result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_exp(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 3;
    float input[] = {0, 1, 2};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_exp_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    TEST_ASSERT_NEAR(output[0], 1.0f, 1e-5f, "exp(0)");
    TEST_ASSERT_NEAR(output[1], 2.718f, 0.01f, "exp(1)");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_sqrt(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 4;
    float input[] = {0, 1, 4, 9};
    float expected[] = {0, 1, 2, 3};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_sqrt_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], expected[i], 1e-5f, "sqrt result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_square(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 5;
    float input[] = {0, 1, 2, 3, 4};
    float expected[] = {0, 1, 4, 9, 16};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_square(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], expected[i], 1e-5f, "square result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内核测试 - 数据操作
 * ============================================================================ */

static int test_scale(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 10;
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float alpha = 2.5f;
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, &alpha, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], input[i] * alpha, 1e-5f, "scale result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_fill(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 10;
    float val = 3.14f;
    float output[N];

    ace_buffer_t buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);

    int n = N;
    void* args[] = {&n, &val, buf_out};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_fill(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], val, 1e-5f, "fill result");
    }

    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_copy(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 10;
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_copy(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], input[i], 1e-5f, "copy result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

static int test_negate(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    const int N = 5;
    float input[] = {-5, -2, 0, 2, 5};
    float expected[] = {5, 2, 0, -2, -5};
    float output[N];

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_negate(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        TEST_ASSERT_NEAR(output[i], expected[i], 1e-5f, "negate result");
    }

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 内核测试 - 线性代数
 * ============================================================================ */

static int test_gemm(void) {
    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CPU, 0, &dev);

    /* 2x3 * 3x2 = 2x2 */
    int M = 2, K = 3, N = 2;
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {7, 8, 9, 10, 11, 12};
    float C[4];
    float expected[] = {58, 64, 139, 154};  /* 手工计算结果 */

    ace_buffer_t buf_A, buf_B, buf_C;
    ace_buffer_alloc(dev, M * K * sizeof(float), &buf_A);
    ace_buffer_alloc(dev, K * N * sizeof(float), &buf_B);
    ace_buffer_alloc(dev, M * N * sizeof(float), &buf_C);

    ace_buffer_write(buf_A, A, M * K * sizeof(float));
    ace_buffer_write(buf_B, B, K * N * sizeof(float));

    void* args[] = {&M, &N, &K, buf_A, buf_B, buf_C};
    int types[] = {ACE_VAL, ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    ace_kernel_invoke(dev, _ace_get_gemm(), ACE_DTYPE_FLOAT32, M, args, types, 6);
    ace_finish(dev);
    ace_buffer_read(buf_C, C, M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(C[i], expected[i], 1e-4f, "GEMM result");
    }

    ace_buffer_free(buf_A);
    ace_buffer_free(buf_B);
    ace_buffer_free(buf_C);
    ace_device_release(dev);
    return 1;
}

/* ============================================================================
 * 多设备测试
 * ============================================================================ */

static int test_multi_device_parallel(void) {
    ace_device_list_t devices;
    ace_error_t err = ace_device_get_all(&devices);

    TEST_ASSERT(err == ACE_OK, "Get all devices");
    TEST_ASSERT(devices.count >= 1, "At least 1 device");

    const int N = 1000;  /* 减少数据量 */
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    ace_sharded_buffer_t sh_a, sh_b, sh_c;
    ace_buffer_alloc_sharded(&devices, N * sizeof(float), &sh_a);
    ace_buffer_alloc_sharded(&devices, N * sizeof(float), &sh_b);
    ace_buffer_alloc_sharded(&devices, N * sizeof(float), &sh_c);

    ace_buffer_write_sharded(&sh_a, a, N * sizeof(float));
    ace_buffer_write_sharded(&sh_b, b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, &sh_a, &sh_b, &sh_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    err = ace_kernel_invoke_sharded(&devices, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    TEST_ASSERT(err == ACE_OK, "Multi-device kernel invoke");
    ace_finish_all(&devices);

    ace_buffer_read_sharded(&sh_c, c, N * sizeof(float));

    /* 验证前 10 个结果 */
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_NEAR(c[i], a[i] + b[i], 1e-5f, "Multi-device result");
    }

    free(a); free(b); free(c);
    ace_buffer_free_sharded(&sh_a);
    ace_buffer_free_sharded(&sh_b);
    ace_buffer_free_sharded(&sh_c);
    ace_device_list_release(&devices);
    return 1;
}

/* ============================================================================
 * 错误处理测试
 * ============================================================================ */

static int test_error_handling(void) {
    /* 测试空设备 */
    ace_error_t err = ace_device_count(ACE_DEVICE_CPU, NULL);
    TEST_ASSERT(err != ACE_OK, "NULL count pointer");

    /* 测试无效设备索引 */
    ace_device_t dev;
    err = ace_device_get(ACE_DEVICE_CPU, 999, &dev);
    TEST_ASSERT(err != ACE_OK, "Invalid device index");

    /* 测试空缓冲区读取 */
    err = ace_buffer_read(NULL, NULL, 0);
    TEST_ASSERT(err != ACE_OK, "NULL buffer read");

    return 1;
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute Unit Tests\n");
    printf("========================================\n\n");

    /* 设备管理测试 */
    printf("--- Device Management Tests ---\n");
    RUN_TEST(test_device_count);
    RUN_TEST(test_device_get_all);
    RUN_TEST(test_device_select_best);
    RUN_TEST(test_device_props);

    /* 内存管理测试 */
    printf("\n--- Memory Management Tests ---\n");
    RUN_TEST(test_buffer_alloc_free);
    RUN_TEST(test_buffer_write_read);

    /* 向量运算测试 */
    printf("\n--- Vector Operation Tests ---\n");
    RUN_TEST(test_vec_add_float);
    RUN_TEST(test_vec_sub_float);
    RUN_TEST(test_vec_mul_float);

    /* 激活函数测试 */
    printf("\n--- Activation Function Tests ---\n");
    RUN_TEST(test_relu);
    RUN_TEST(test_sigmoid);
    RUN_TEST(test_tanh);

    /* 数学函数测试 */
    printf("\n--- Math Function Tests ---\n");
    RUN_TEST(test_abs);
    RUN_TEST(test_exp);
    RUN_TEST(test_sqrt);
    RUN_TEST(test_square);

    /* 数据操作测试 */
    printf("\n--- Data Operation Tests ---\n");
    RUN_TEST(test_scale);
    RUN_TEST(test_fill);
    RUN_TEST(test_copy);
    RUN_TEST(test_negate);

    /* 线性代数测试 */
    printf("\n--- Linear Algebra Tests ---\n");
    RUN_TEST(test_gemm);

    /* 多设备测试 */
    printf("\n--- Multi-Device Tests ---\n");
    RUN_TEST(test_multi_device_parallel);

    /* 错误处理测试 */
    printf("\n--- Error Handling Tests ---\n");
    RUN_TEST(test_error_handling);

    /* 总结 */
    printf("\n========================================\n");
    printf("  Tests: %d/%d passed\n", g_tests_passed, g_tests_run);
    printf("========================================\n");

    return g_tests_passed == g_tests_run ? 0 : 1;
}
