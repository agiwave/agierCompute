/**
 * @file comprehensive_test.c
 * @brief AgierCompute 综合测试套件
 * 
 * 测试所有后端和所有预置内核
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* 内核定义 */
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

ACE_KERNEL(abs_kernel,
    void abs_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] < 0 ? -in[i] : in[i];
    }
);

ACE_KERNEL(sqrt_kernel,
    void sqrt_kernel(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = sqrtf(in[i]);
    }
);

/* 测试配置 */
typedef struct {
    const char* name;
    ace_device_type_t type;
    int device_index;
} test_config_t;

static int g_tests_passed = 0;
static int g_tests_total = 0;

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); \
        return 0; \
    } \
} while(0)

#define RUN_TEST_ON_DEVICE(test_fn, dev) do { \
    g_tests_total++; \
    if (test_fn(dev)) { \
        g_tests_passed++; \
        printf("  PASS\n"); \
    } else { \
        printf("  FAIL\n"); \
    } \
} while(0)

/* 测试函数 */
static int test_vec_add(ace_device_t dev) {
    const int N = 100;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);
    
    ace_buffer_write(buf_a, a, N * sizeof(float));
    ace_buffer_write(buf_b, b, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(buf_c, c, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(c[i], a[i] + b[i], 1e-5f, "vec_add");
    }
    
    free(a); free(b); free(c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    return 1;
}

static int test_vec_sub(ace_device_t dev) {
    const int N = 100;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) { a[i] = i * 3; b[i] = i; }
    
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
        ASSERT_NEAR(c[i], a[i] - b[i], 1e-5f, "vec_sub");
    }
    
    free(a); free(b); free(c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    return 1;
}

static int test_vec_mul(ace_device_t dev) {
    const int N = 100;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) { a[i] = i * 0.5f; b[i] = i * 2.0f; }
    
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
        ASSERT_NEAR(c[i], a[i] * b[i], 1e-4f, "vec_mul");
    }
    
    free(a); free(b); free(c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    return 1;
}

static int test_scale(ace_device_t dev) {
    const int N = 100;
    float *in = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));
    float alpha = 2.5f;
    
    for (int i = 0; i < N; i++) in[i] = i * 0.1f;
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, &alpha, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(buf_out, out, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(out[i], in[i] * alpha, 1e-5f, "scale");
    }
    
    free(in); free(out);
    ace_buffer_free(buf_in); ace_buffer_free(buf_out);
    return 1;
}

static int test_relu(ace_device_t dev) {
    const int N = 10;
    float input[] = {-5, -2, -1, 0, 1, 2, 3, 4, 5, 10};
    float expected[] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 10};
    float *out = malloc(N * sizeof(float));
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_relu(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f, "relu");
    }
    
    free(out);
    ace_buffer_free(buf_in); ace_buffer_free(buf_out);
    return 1;
}

static int test_sigmoid(ace_device_t dev) {
    const int N = 5;
    float input[] = {-2, -1, 0, 1, 2};
    float *out = malloc(N * sizeof(float));
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_sigmoid(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, out, N * sizeof(float));
    
    /* sigmoid(0) = 0.5 */
    ASSERT_NEAR(out[2], 0.5f, 1e-5f, "sigmoid(0)");
    /* sigmoid(1) ≈ 0.731 */
    ASSERT_NEAR(out[3], 0.731f, 0.01f, "sigmoid(1)");
    
    free(out);
    ace_buffer_free(buf_in); ace_buffer_free(buf_out);
    return 1;
}

static int test_abs(ace_device_t dev) {
    const int N = 6;
    float input[] = {-5, -2, 0, 2, 5, 10};
    float expected[] = {5, 2, 0, 2, 5, 10};
    float *out = malloc(N * sizeof(float));
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_abs_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f, "abs");
    }
    
    free(out);
    ace_buffer_free(buf_in); ace_buffer_free(buf_out);
    return 1;
}

static int test_sqrt(ace_device_t dev) {
    const int N = 5;
    float input[] = {0, 1, 4, 9, 16};
    float expected[] = {0, 1, 2, 3, 4};
    float *out = malloc(N * sizeof(float));
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, input, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_sqrt_kernel(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(buf_out, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f, "sqrt");
    }
    
    free(out);
    ace_buffer_free(buf_in); ace_buffer_free(buf_out);
    return 1;
}

/* 运行单个设备的完整测试 */
static void run_device_tests(const char* backend_name, ace_device_t dev) {
    ace_device_props_t props;
    ace_device_props(dev, &props);
    
    printf("\n=== Testing %s: %s ===\n", backend_name, props.name);
    
    printf("  vec_add... ");
    RUN_TEST_ON_DEVICE(test_vec_add, dev);
    
    printf("  vec_sub... ");
    RUN_TEST_ON_DEVICE(test_vec_sub, dev);
    
    printf("  vec_mul... ");
    RUN_TEST_ON_DEVICE(test_vec_mul, dev);
    
    printf("  scale... ");
    RUN_TEST_ON_DEVICE(test_scale, dev);
    
    printf("  relu... ");
    RUN_TEST_ON_DEVICE(test_relu, dev);
    
    printf("  sigmoid... ");
    RUN_TEST_ON_DEVICE(test_sigmoid, dev);
    
    printf("  abs... ");
    RUN_TEST_ON_DEVICE(test_abs, dev);
    
    printf("  sqrt... ");
    RUN_TEST_ON_DEVICE(test_sqrt, dev);
    
    ace_device_release(dev);
}

int main() {
    printf("========================================\n");
    printf("  AgierCompute Comprehensive Tests\n");
    printf("========================================\n");
    
    /* 测试每个可用的后端 */
    ace_device_type_t types[] = {
        ACE_DEVICE_CPU,
        ACE_DEVICE_CUDA,
        ACE_DEVICE_OPENCL,
        ACE_DEVICE_VULKAN
    };
    const char* names[] = {"CPU", "CUDA", "OpenCL", "Vulkan"};
    int num_types = sizeof(types) / sizeof(types[0]);
    
    for (int i = 0; i < num_types; i++) {
        int count = 0;
        ace_device_count(types[i], &count);
        
        if (count > 0) {
            ace_device_t dev;
            if (ace_device_get(types[i], 0, &dev) == ACE_OK && dev) {
                run_device_tests(names[i], dev);
            }
        } else {
            printf("\n=== %s: No devices available ===\n", names[i]);
        }
    }
    
    printf("\n========================================\n");
    printf("  Results: %d/%d tests passed\n", g_tests_passed, g_tests_total);
    printf("========================================\n");
    
    return (g_tests_passed == g_tests_total) ? 0 : 1;
}
