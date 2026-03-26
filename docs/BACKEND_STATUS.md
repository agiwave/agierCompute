# AgierCompute 后端状态

## 后端实现状态

| 后端 | 状态 | 说明 |
|------|------|------|
| CUDA | ✅ 完整 | NVRTC 运行时编译 |
| OpenCL | ✅ 完整 | 运行时编译 |
| Vulkan | ✅ 完整 | SPIR-V 编译（需 shaderc） |
| CPU | ❌ 未实现 | 占位实现，无内核执行能力 |
| Metal | ❌ 未实现 | 未实现 |

## 数据类型支持矩阵

| 数据类型 | 大小 | CUDA | OpenCL | Vulkan | 说明 |
|----------|------|------|--------|--------|------|
| **FLOAT32** | 4B | ✅ | ✅ | ✅ | 单精度浮点 - 完全支持 |
| **FLOAT64** | 8B | ✅ | ✅ | ✅ | 双精度浮点 - 完全支持 |
| **INT32** | 4B | ✅ | ✅ | ✅ | 32 位整数 - 完全支持 |
| **INT64** | 8B | ✅ | ✅ | ⚠️ | 64 位整数 - Vulkan 回退到 32 位 |
| **FLOAT16** | 2B | ✅ | ✅ | ✅ | 半精度浮点 - AI 推理常用 |
| **BFLOAT16** | 2B | ✅ | ✅ | ✅ | Brain 浮点 - AI 训练常用 |
| **INT8** | 1B | ✅ | ✅ | ✅ | 8 位整数 - 量化常用 |
| **UINT8** | 1B | ✅ | ✅ | ✅ | 8 位无符号整数 |
| **INT16** | 2B | ✅ | ✅ | ✅ | 16 位整数 |
| **BOOL** | 1B | ⚠️ | ⚠️ | ⚠️ | 布尔值 - 有限支持 |

### 数据类型说明

#### 浮点类型
- **FLOAT32**: 标准单精度浮点，所有后端完全支持
- **FLOAT64**: 标准双精度浮点，需要设备支持 FP64（某些移动 GPU 可能性能较低）
- **FLOAT16**: IEEE 754 半精度浮点
  - CUDA: 使用 `half` 类型，需要 compute capability >= 6.0
  - OpenCL: 使用 `half` 类型，需要 `cl_khr_fp16` 扩展
  - Vulkan: 使用 `float16_t` 类型，需要 `GL_EXT_shader_explicit_arithmetic_types_float16`
- **BFLOAT16**: Google Brain 浮点格式（1 符号位 +8 指数位 +7 尾数位）
  - CUDA: 使用 `__nv_bfloat16` 类型（CUDA 11+）
  - OpenCL: 使用 `ushort` 存储，提供转换宏
  - Vulkan: 使用 `int16_t` 存储，提供转换宏

#### 整数类型
- **INT8/UINT8**: 8 位整数，常用于量化神经网络
- **INT16**: 16 位整数
- **INT32**: 标准 32 位整数
- **INT64**: 64 位整数
  - CUDA/OpenCL: 原生支持 64 位整数
  - Vulkan: 回退到 32 位 int（大多数设备不支持 64 位扩展）
  - 注意：在 Vulkan 上使用 INT64 可能导致溢出，对于大整数请使用 INT32

## 后端状态详情

### CUDA ✅
- **编译方式**: NVRTC 运行时编译
- **依赖**: NVIDIA CUDA Toolkit
- **特性**: 
  - 完整的内核编译和执行支持
  - FLOAT16/BFLOAT16 原生支持（compute >= 6.0）
  - 自动启用 `-use_fast_math` 优化 FP16 性能
- **最低要求**: CUDA Compute Capability 6.0+ (推荐 FP16 支持)

### OpenCL ✅
- **编译方式**: 运行时编译
- **依赖**: OpenCL SDK
- **特性**: 
  - 跨平台 GPU/CPU 支持
  - FLOAT16 通过 `cl_khr_fp16` 扩展支持
  - BFLOAT16 使用 `ushort` 存储 + 转换宏
- **注意**: 某些设备可能不支持 `cl_khr_fp16` 扩展

### Vulkan ✅
- **编译方式**: SPIR-V 编译（shaderc）
- **依赖**: Vulkan SDK + shaderc
- **特性**: 
  - 支持所有 Vulkan 设备
  - FLOAT16 使用 `float16_t` 内置类型
  - BFLOAT16 使用 `int16_t` 存储 + 转换宏
- **扩展要求**:
  - FLOAT16: `GL_EXT_shader_explicit_arithmetic_types_float16`
  - BFLOAT16/INT16: `GL_EXT_shader_explicit_arithmetic_types_int16`
  - INT64: 大多数设备不支持 64 位扩展，回退到 32 位 int

### CPU ❌
- **状态**: 占位实现
- **设备数量**: 0
- **缺失**: 内核编译和执行（需要 GCC JIT/TCC/LLVM）

## 测试状态

### 后端测试
- CUDA: ✅ 通过
- OpenCL: ✅ 通过
- Vulkan: ✅ 通过

### 数据类型测试
- FLOAT32/FLOAT64/INT32: ✅ 所有后端通过
- INT64: ✅ CUDA/OpenCL 完全支持，Vulkan 回退到 32 位
- INT8/UINT8/INT16: ✅ 所有后端通过
- FLOAT16/BFLOAT16: ✅ 已实现支持（需要设备扩展）

### 功能测试
- 向量加法/乘法：✅ 所有后端通过
- 标量乘法：✅ 所有后端通过
- 激活函数 (ReLU/Sigmoid): ✅ 所有后端通过

## 性能基准

运行基准测试：
```bash
cd build && ./bin/benchmark
```

测试项目：
- 向量加法带宽
- 向量乘法带宽

结果保存为 CSV：
```c
ace_benchmark_save_csv(results, count, "benchmark.csv");
```

## 使用示例

### FLOAT16 使用示例
```c
#include "ace.h"

// 定义内核（使用泛型 T）
ACE_KERNEL(vec_add_fp16,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

// 执行 FLOAT16 内核
void* args[] = {&n, buf_a, buf_b, buf_c};
int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
ace_kernel_invoke(dev, _ace_get_vec_add_fp16(), 
                  ACE_DTYPE_FLOAT16, N, args, types, 4);
```

### BFLOAT16 注意事项
BFLOAT16 在 OpenCL 和 Vulkan 中使用整数存储，建议在内核中使用提供的宏：
```c
ACE_KERNEL(bf16_add,
    void bf16_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) {
            // 转换为 float 计算，再转回 bfloat16
            c[i] = BF16_ADD(a[i], b[i]);
        }
    }
);
```
