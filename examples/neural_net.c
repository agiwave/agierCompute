/**
 * @file neural_net.c
 * @brief 使用 AgierCompute 实现简单的两层神经网络前向传播
 *
 * 网络结构：Input -> FC1 -> ReLU -> FC2 -> Softmax -> Output
 *
 * 演示：
 * - 使用计算原语构建神经网络
 * - 内存池优化内存分配
 * - Stream 异步执行
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义（必须在函数外）
 * ============================================================================ */

ACE_KERNEL(fc_kernel,
    void fc(int out_size, int in_size, T* W, T* b, T* x, T* y) {
        int out_id = GID;
        if (out_id < out_size) {
            T sum = 0;
            for (int i = 0; i < in_size; i++) {
                sum += W[out_id * in_size + i] * x[i];
            }
            y[out_id] = sum + b[out_id];
        }
    }
);

ACE_KERNEL(softmax_kernel,
    void softmax(int n, T* x, T* y) {
        int i = GID;
        if (i < n) {
            T exp_val = expf(x[i]);
            y[i] = exp_val;
        }
    }
);

/* ============================================================================
 * 简单的神经网络结构
 * ============================================================================ */

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    /* 权重和偏置 */
    ace_buffer_t W1;  /* [hidden_size, input_size] */
    ace_buffer_t b1;  /* [hidden_size] */
    ace_buffer_t W2;  /* [output_size, hidden_size] */
    ace_buffer_t b2;  /* [output_size] */

    /* 中间结果 */
    ace_buffer_t z1;  /* 第一层输出 (before ReLU) */
    ace_buffer_t a1;  /* 第一层输出 (after ReLU) */
    ace_buffer_t z2;  /* 第二层输出 (before Softmax) */
    ace_buffer_t a2;  /* 最终输出 (after Softmax) */

    ace_device_t dev;
    ace_mempool_t pool;
    ace_stream_t stream;
} neural_net_t;

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/* 初始化权重 (Xavier 初始化) */
static void init_weights(float* W, int rows, int cols) {
    float scale = sqrtf(2.0f / (rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        W[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}

/* 初始化偏置为零 */
static void init_bias(float* b, int size) {
    for (int i = 0; i < size; i++) {
        b[i] = 0.0f;
    }
}

/* ============================================================================
 * 神经网络 API
 * ============================================================================ */

int neural_net_init(neural_net_t* net, int input_size, int hidden_size, int output_size) {
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;

    /* 使用 CPU 后端（更稳定） */
    ace_error_t err = ace_device_get(ACE_DEVICE_CPU, 0, &net->dev);
    if (err != ACE_OK) {
        printf("Failed to get CPU device\n");
        return -1;
    }

    /* 创建内存池 */
    net->pool = ace_mempool_create(net->dev);
    if (!net->pool) {
        printf("Failed to create memory pool\n");
        return -1;
    }

    /* 创建流 */
    err = ace_stream_create(net->dev, &net->stream);
    if (err != ACE_OK) {
        printf("Failed to create stream\n");
        return -1;
    }

    /* 分配权重和偏置 */
    size_t W1_size = hidden_size * input_size * sizeof(float);
    size_t b1_size = hidden_size * sizeof(float);
    size_t W2_size = output_size * hidden_size * sizeof(float);
    size_t b2_size = output_size * sizeof(float);

    ace_mempool_alloc(net->pool, W1_size, &net->W1);
    ace_mempool_alloc(net->pool, b1_size, &net->b1);
    ace_mempool_alloc(net->pool, W2_size, &net->W2);
    ace_mempool_alloc(net->pool, b2_size, &net->b2);

    /* 分配中间结果 */
    size_t z1_size = hidden_size * sizeof(float);
    size_t a1_size = hidden_size * sizeof(float);
    size_t z2_size = output_size * sizeof(float);
    size_t a2_size = output_size * sizeof(float);

    ace_mempool_alloc(net->pool, z1_size, &net->z1);
    ace_mempool_alloc(net->pool, a1_size, &net->a1);
    ace_mempool_alloc(net->pool, z2_size, &net->z2);
    ace_mempool_alloc(net->pool, a2_size, &net->a2);

    /* 初始化权重 */
    float* h_W1 = (float*)malloc(W1_size);
    float* h_b1 = (float*)malloc(b1_size);
    float* h_W2 = (float*)malloc(W2_size);
    float* h_b2 = (float*)malloc(b2_size);

    init_weights(h_W1, hidden_size, input_size);
    init_bias(h_b1, hidden_size);
    init_weights(h_W2, output_size, hidden_size);
    init_bias(h_b2, output_size);

    /* 写入设备 */
    ace_stream_memcpy_h2d(net->stream, net->W1, h_W1, W1_size);
    ace_stream_memcpy_h2d(net->stream, net->b1, h_b1, b1_size);
    ace_stream_memcpy_h2d(net->stream, net->W2, h_W2, W2_size);
    ace_stream_memcpy_h2d(net->stream, net->b2, h_b2, b2_size);
    ace_stream_synchronize(net->stream);

    free(h_W1); free(h_b1); free(h_W2); free(h_b2);

    printf("Neural net initialized:\n");
    printf("  Input:  %d\n", input_size);
    printf("  Hidden: %d\n", hidden_size);
    printf("  Output: %d\n", output_size);

    return 0;
}

void neural_net_destroy(neural_net_t* net) {
    if (!net) return;

    /* 释放所有缓冲区 */
    ace_mempool_free(net->pool, net->W1);
    ace_mempool_free(net->pool, net->b1);
    ace_mempool_free(net->pool, net->W2);
    ace_mempool_free(net->pool, net->b2);
    ace_mempool_free(net->pool, net->z1);
    ace_mempool_free(net->pool, net->a1);
    ace_mempool_free(net->pool, net->z2);
    ace_mempool_free(net->pool, net->a2);

    ace_stream_destroy(net->stream);
    ace_mempool_destroy(net->pool);
    ace_device_release(net->dev);
}

/* 前向传播 */
ace_error_t neural_net_forward(neural_net_t* net, float* input, float* output) {
    /* 首先将输入写入设备 */
    ace_buffer_t input_buf;
    ace_mempool_alloc(net->pool, net->input_size * sizeof(float), &input_buf);
    ace_buffer_write(input_buf, input, net->input_size * sizeof(float));

    /* 层 1: z1 = W1 * input + b1 */
    {
        int out_size = net->hidden_size;
        int in_size = net->input_size;
        void* args[] = {&out_size, &in_size, net->W1, net->b1, input_buf, net->z1};
        int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF, ACE_BUF};
        ace_stream_launch(net->stream, _ace_get_fc_kernel(), ACE_DTYPE_FLOAT32, out_size, args, types, 6);
    }

    /* 层 1: a1 = ReLU(z1) */
    ace_relu(net->stream, net->hidden_size, net->z1, net->a1);

    /* 层 2: z2 = W2 * a1 + b2 */
    {
        int out_size = net->output_size;
        int in_size = net->hidden_size;
        void* args[] = {&out_size, &in_size, net->W2, net->b2, net->a1, net->z2};
        int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF, ACE_BUF};
        ace_stream_launch(net->stream, _ace_get_fc_kernel(), ACE_DTYPE_FLOAT32, out_size, args, types, 6);
    }

    /* 层 2: a2 = Softmax(z2) - 先计算 exp */
    {
        int n = net->output_size;
        void* args[] = {&n, net->z2, net->a2};
        int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
        ace_stream_launch(net->stream, _ace_get_softmax_kernel(), ACE_DTYPE_FLOAT32, n, args, types, 3);
    }

    ace_stream_synchronize(net->stream);

    /* 调试：查看中间结果 */
    float* h_z1 = (float*)malloc(net->hidden_size * sizeof(float));
    ace_buffer_read(net->z1, h_z1, net->hidden_size * sizeof(float));
    printf("Debug - z1[0:5]: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_z1[i]);
    printf("\n");
    free(h_z1);

    /* 读取 z2 并进行数值稳定的 softmax */
    float* h_z2 = (float*)malloc(net->output_size * sizeof(float));
    ace_buffer_read(net->z2, h_z2, net->output_size * sizeof(float));

    printf("Debug - z2[0:5]: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_z2[i]);
    printf("\n");

    /* 找到最大值 */
    float max_val = h_z2[0];
    for (int i = 1; i < net->output_size; i++) {
        if (h_z2[i] > max_val) max_val = h_z2[i];
    }

    /* 减去最大值并计算 exp */
    float sum = 0;
    for (int i = 0; i < net->output_size; i++) {
        h_z2[i] = expf(h_z2[i] - max_val);
        sum += h_z2[i];
    }

    /* 归一化 */
    for (int i = 0; i < net->output_size; i++) {
        h_z2[i] /= sum;
    }

    if (output) {
        memcpy(output, h_z2, net->output_size * sizeof(float));
    }

    free(h_z2);
    ace_mempool_free(net->pool, input_buf);

    return ACE_OK;
}

/* 打印网络输出 */
void neural_net_print_output(neural_net_t* net) {
    float* h_output = (float*)malloc(net->output_size * sizeof(float));
    ace_buffer_read(net->a2, h_output, net->output_size * sizeof(float));

    printf("Output probabilities:\n");
    for (int i = 0; i < net->output_size; i++) {
        printf("  Class %d: %.4f\n", i, h_output[i]);
    }

    /* 找到预测类别 */
    int predicted = 0;
    float max_prob = h_output[0];
    for (int i = 1; i < net->output_size; i++) {
        if (h_output[i] > max_prob) {
            max_prob = h_output[i];
            predicted = i;
        }
    }
    printf("Predicted class: %d (confidence: %.2f%%)\n", predicted, max_prob * 100);

    free(h_output);
}

/* ============================================================================
 * 主函数 - 演示神经网络推理
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Neural Network Demo\n");
    printf("========================================\n\n");

    /* 创建神经网络：10 -> 32 -> 10 */
    neural_net_t net;
    if (neural_net_init(&net, 10, 32, 10) != 0) {
        printf("Failed to initialize neural network\n");
        return 1;
    }

    /* 创建随机输入 */
    float input[10];
    for (int i = 0; i < 10; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    }

    printf("\nInput vector:\n");
    printf("  ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", input[i]);
    }
    printf("\n\n");

    /* 前向传播 */
    float output[10];
    printf("Running forward pass...\n");
    neural_net_forward(&net, input, output);

    /* 打印结果 */
    neural_net_print_output(&net);

    /* 清理 */
    neural_net_destroy(&net);

    printf("\n========================================\n");
    printf("  Demo completed!\n");
    printf("========================================\n");

    return 0;
}
