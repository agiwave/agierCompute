/**
 * @file simple_test.c
 * @brief 测试简化的 CUDA 风格 API 和 JIT 编译
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

/* 定义内核 */
ACE_KERNEL(vec_add,
    void(int n, float* a, float* b, float* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

ACE_KERNEL(vec_scale,
    void(int n, float alpha, float* in, float* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
);

ACE_KERNEL(relu,
    void(int n, float* in, float* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
);

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Simple Test\n");
    printf("========================================\n\n");

    /* 设置设备 */
    ace_set_device(ACE_DEVICE_CPU, 0);
    ace_print_device();

    /* 测试向量加法 */
    printf("\n--- Test: vec_add ---\n");
    {
        const int N = 10;
        float h_a[N], h_b[N], h_c[N];
        void *d_a, *d_b, *d_c;

        for (int i = 0; i < N; i++) {
            h_a[i] = i * 1.0f;
            h_b[i] = i * 2.0f;
        }

        ace_malloc(&d_a, N * sizeof(float));
        ace_malloc(&d_b, N * sizeof(float));
        ace_malloc(&d_c, N * sizeof(float));

        ace_memcpy_h2d(d_a, h_a, N * sizeof(float));
        ace_memcpy_h2d(d_b, h_b, N * sizeof(float));

        ace_launch(ace_kernel_vec_add(), N, "ippp", N, d_a, d_b, d_c);
        ace_sync();

        ace_memcpy_d2h(h_c, d_c, N * sizeof(float));

        printf("  a: ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_a[i]);
        printf("\n  b: ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_b[i]);
        printf("\n  c=a+b: ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_c[i]);
        printf("\n");

        int pass = 1;
        for (int i = 0; i < N; i++) {
            if (h_c[i] != h_a[i] + h_b[i]) pass = 0;
        }
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");

        ace_free(d_a);
        ace_free(d_b);
        ace_free(d_c);
    }

    /* 测试向量缩放 */
    printf("\n--- Test: vec_scale ---\n");
    {
        const int N = 5;
        float h_in[N], h_out[N];
        void *d_in, *d_out;
        float alpha = 2.5f;

        for (int i = 0; i < N; i++) h_in[i] = i * 1.0f;

        ace_malloc(&d_in, N * sizeof(float));
        ace_malloc(&d_out, N * sizeof(float));

        ace_memcpy_h2d(d_in, h_in, N * sizeof(float));

        ace_launch(ace_kernel_vec_scale(), N, "ifpp", N, alpha, d_in, d_out);
        ace_sync();

        ace_memcpy_d2h(h_out, d_out, N * sizeof(float));

        printf("  in: ");
        for (int i = 0; i < N; i++) printf("%.1f ", h_in[i]);
        printf("\n  alpha: %.1f\n", alpha);
        printf("  out=in*alpha: ");
        for (int i = 0; i < N; i++) printf("%.1f ", h_out[i]);
        printf("\n");

        int pass = 1;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != h_in[i] * alpha) pass = 0;
        }
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");

        ace_free(d_in);
        ace_free(d_out);
    }

    /* 测试 ReLU */
    printf("\n--- Test: relu ---\n");
    {
        const int N = 6;
        float h_in[] = {-3, -1, 0, 1, 2, 5};
        float h_out[N];
        void *d_in, *d_out;

        ace_malloc(&d_in, N * sizeof(float));
        ace_malloc(&d_out, N * sizeof(float));

        ace_memcpy_h2d(d_in, h_in, N * sizeof(float));

        ace_launch(ace_kernel_relu(), N, "pp", d_in, d_out);
        ace_sync();

        ace_memcpy_d2h(h_out, d_out, N * sizeof(float));

        printf("  in:  ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_in[i]);
        printf("\n  out: ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_out[i]);
        printf("\n");

        float expected[] = {0, 0, 0, 1, 2, 5};
        int pass = 1;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != expected[i]) pass = 0;
        }
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");

        ace_free(d_in);
        ace_free(d_out);
    }

    printf("\n========================================\n");
    printf("  Test completed!\n");
    printf("========================================\n");

    return 0;
}
