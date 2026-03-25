/**
 * @file user_kernels.c
 * @brief 用户内核示例 - 展示如何使用AgierCompute编写自己的内核
 * 
 * 这是用户编写的内核代码，使用ACE语法。
 * 引擎会将其翻译到CPU/CUDA/OpenCL等平台。
 */

#include "ace.h"

/* ============================================================================
 * 示例1: 向量加法
 * ============================================================================ */

const char* kernel_vec_add = R"(
ACE_KERNEL(vec_add, (int n, ACE_GLOBAL float* a, ACE_GLOBAL float* b, ACE_GLOBAL float* c))
{
    int i = ACE_GLOBAL_ID;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)";

/* ============================================================================
 * 示例2: ReLU激活函数
 * ============================================================================ */

const char* kernel_relu = R"(
ACE_KERNEL(relu, (int n, ACE_GLOBAL float* x, ACE_GLOBAL float* y))
{
    int i = ACE_GLOBAL_ID;
    if (i < n) {
        float val = x[i];
        y[i] = val > 0.0f ? val : 0.0f;
    }
}
)";

/* ============================================================================
 * 示例3: 使用内建数学函数
 * ============================================================================ */

const char* kernel_sigmoid = R"(
ACE_KERNEL(sigmoid, (int n, ACE_GLOBAL float* x, ACE_GLOBAL float* y))
{
    int i = ACE_GLOBAL_ID;
    if (i < n) {
        y[i] = 1.0f / (1.0f + ace_exp(-x[i]));
    }
}
)";

/* ============================================================================
 * 示例4: 矩阵乘法 (分块算法)
 * ============================================================================ */

const char* kernel_matmul = R"(
ACE_KERNEL(matmul, 
    (int M, int N, int K,
     ACE_GLOBAL float* A, ACE_GLOBAL float* B, ACE_GLOBAL float* C))
{
    int row = ACE_GLOBAL_ID_X;
    int col = ACE_GLOBAL_ID_Y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";

/* ============================================================================
 * 示例5: 归约求和 (使用共享内存)
 * ============================================================================ */

const char* kernel_reduce_sum = R"(
ACE_KERNEL(reduce_sum, (int n, ACE_GLOBAL float* input, ACE_GLOBAL float* output))
{
    ACE_SHARED float shared[256];
    
    int tid = ACE_LOCAL_ID;
    int gid = ACE_GLOBAL_ID;
    int block_size = ACE_GROUP_SIZE;
    
    /* 加载数据到共享内存 */
    shared[tid] = (gid < n) ? input[gid] : 0.0f;
    ACE_BARRIER();
    
    /* 树形归约 */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        ACE_BARRIER();
    }
    
    /* 写回结果 */
    if (tid == 0) {
        output[ACE_GROUP_ID] = shared[0];
    }
}
)";

/* ============================================================================
 * 示例6: Softmax
 * ============================================================================ */

const char* kernel_softmax = R"(
ACE_KERNEL(softmax, (int batch_size, int seq_len, ACE_GLOBAL float* input, ACE_GLOBAL float* output))
{
    int batch = ACE_GLOBAL_ID;
    
    if (batch < batch_size) {
        float* x = input + batch * seq_len;
        float* y = output + batch * seq_len;
        
        /* 找最大值 */
        float max_val = x[0];
        for (int i = 1; i < seq_len; i++) {
            max_val = ace_max(max_val, x[i]);
        }
        
        /* 计算exp并求和 */
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            y[i] = ace_exp(x[i] - max_val);
            sum += y[i];
        }
        
        /* 归一化 */
        for (int i = 0; i < seq_len; i++) {
            y[i] /= sum;
        }
    }
}
)";

/* ============================================================================
 * 示例7: Layer Normalization
 * ============================================================================ */

const char* kernel_layer_norm = R"(
ACE_KERNEL(layer_norm, 
    (int batch_size, int hidden_size, float eps,
     ACE_GLOBAL float* x, ACE_GLOBAL float* gamma, ACE_GLOBAL float* beta, ACE_GLOBAL float* y))
{
    int batch = ACE_GLOBAL_ID;
    
    if (batch < batch_size) {
        float* px = x + batch * hidden_size;
        float* py = y + batch * hidden_size;
        
        /* 计算均值 */
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += px[i];
        }
        mean /= hidden_size;
        
        /* 计算方差 */
        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = px[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        /* 归一化 */
        float inv_std = 1.0f / ace_sqrt(var + eps);
        for (int i = 0; i < hidden_size; i++) {
            py[i] = (px[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}
)";

/* ============================================================================
 * 示例8: GELU激活函数
 * ============================================================================ */

const char* kernel_gelu = R"(
ACE_KERNEL(gelu, (int n, ACE_GLOBAL float* x, ACE_GLOBAL float* y))
{
    int i = ACE_GLOBAL_ID;
    if (i < n) {
        float val = x[i];
        /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
        float c = ace_sqrt(2.0f / 3.14159265358979323846f);
        y[i] = 0.5f * val * (1.0f + ace_tanh(c * (val + 0.044715f * val * val * val)));
    }
}
)";

/* ============================================================================
 * 示例9: 使用原子操作的直方图
 * ============================================================================ */

const char* kernel_histogram = R"(
ACE_KERNEL(histogram, (int n, int num_bins, ACE_GLOBAL float* data, ACE_GLOBAL int* bins))
{
    int i = ACE_GLOBAL_ID;
    
    if (i < n) {
        int bin = (int)(data[i] * num_bins);
        bin = ace_clamp(bin, 0, num_bins - 1);
        ace_atomic_add(&bins[bin], 1);
    }
}
)";