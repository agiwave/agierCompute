# AgierCompute 后端状态

## 后端实现状态

| 后端 | 状态 | 说明 |
|------|------|------|
| CUDA | ✅ 完整 | NVRTC 运行时编译，表驱动架构 |
| OpenCL | ✅ 完整 | 运行时编译，表驱动架构 |
| Vulkan | ✅ 完整 | SPIR-V 编译（需 shaderc），表驱动架构 |
| CPU | ❌ 未实现 | 占位实现，无内核执行能力 |
| Metal | ❌ 未实现 | 未实现 |

## 数据类型支持矩阵

| 数据类型 | 大小 | CUDA | OpenCL | Vulkan | 说明 |
|----------|------|------|--------|--------|------|
| **FLOAT32** | 4B | ✅ | ✅ | ✅ | 单精度浮点 - 完全支持 |
| **FLOAT64** | 8B | ✅ | ✅ | ✅ | 双精度浮点 - 完全支持 |
| **INT32** | 4B | ✅ | ✅ | ✅ | 32 位整数 - 完全支持 |
| **INT64** | 8B | ✅ | ✅ | ✅ | 64 位整数 - 完全支持 |
| **FLOAT16** | 2B | ✅ | ✅ | ✅ | 半精度浮点 - AI 推理常用 |
| **BFLOAT16** | 2B | ✅ | ✅ | ✅ | Brain 浮点 - AI 训练常用 |
| **INT8** | 1B | ✅ | ✅ | ✅ | 8 位整数 - 量化常用 |
| **UINT8** | 1B | ✅ | ✅ | ✅ | 8 位无符号整数 |
| **INT16** | 2B | ✅ | ✅ | ✅ | 16 位整数 |
| **BOOL** | 1B | ⚠️ | ⚠️ | ⚠️ | 布尔值 - 有限支持 |

## 后端架构

### 表驱动设计

所有后端采用统一的表驱动架构：

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
  - 类型名称、扩展、转换函数、宏定义
  - 根据设备能力自动判断是否需要模拟实现
- **内核操作表** (`*_kernel_ops_table.h/c`): 定义所有内核操作
  - 每个操作一个 inject 函数，高内聚实现
  - 根据类型信息自动生成原生或模拟实现

### 架构优势

1. **高内聚**: 每个操作的完整实现在一个 inject 函数内
2. **低耦合**: 类型和操作完全隔离
3. **易扩展**: 添加新类型只需在类型表中添加条目
4. **易维护**: 所有类型/操作相关逻辑集中在表中

## 后端状态详情

### CUDA ✅
- **编译方式**: NVRTC 运行时编译
- **依赖**: NVIDIA CUDA Toolkit
- **表驱动实现**:
  - `cuda_dtype_table.h/c`: 数据类型表
  - `cuda_kernel_ops_table.h/c`: 内核操作表
- **特性**:
  - 完整的内核编译和执行支持
  - FLOAT16/BFLOAT16 使用 unsigned short 模拟实现
  - 自动启用 `-use_fast_math` 优化性能
- **最低要求**: CUDA Compute Capability 6.0+ (推荐 FP16 支持)

### OpenCL ✅
- **编译方式**: 运行时编译
- **依赖**: OpenCL SDK
- **表驱动实现**:
  - `opencl_dtype_table.h/c`: 数据类型表
  - `opencl_kernel_ops_table.h/c`: 内核操作表
- **特性**:
  - 跨平台 GPU/CPU 支持
  - FLOAT16 通过 `cl_khr_fp16` 扩展支持
  - BFLOAT16 使用 `ushort` 存储 + 转换函数
  - 自动为指针参数添加 `__global` 限定符
- **注意**: 某些设备可能不支持 `cl_khr_fp16` 扩展

### Vulkan ✅
- **编译方式**: SPIR-V 编译（shaderc）
- **依赖**: Vulkan SDK + shaderc
- **表驱动实现**:
  - `vulkan_dtype_table.h/c`: 数据类型表
  - `vulkan_kernel_ops_table.h/c`: 内核操作表
- **特性**:
  - 支持所有 Vulkan 设备
  - 设备能力检测（shaderFloat16/shaderInt64 等）
  - 根据设备能力自动选择原生或模拟实现
  - 正确的 push constants 对齐处理
- **扩展要求**:
  - FLOAT16: `GL_EXT_shader_explicit_arithmetic_types_float16`
  - INT64: `GL_EXT_shader_explicit_arithmetic_types_int64`

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
- INT64: ✅ 所有后端完全支持
- INT8/UINT8/INT16: ✅ 所有后端通过
- FLOAT16/BFLOAT16: ✅ 所有后端通过

### 功能测试
- 向量加法/乘法：✅ 所有后端通过
- 标量乘法：✅ 所有后端通过
- 激活函数 (ReLU/Sigmoid): ✅ 所有后端通过
- 比较运算 (MAX/MIN): ✅ 所有后端通过
- 算术运算 (ABS/SQUARE): ✅ 所有后端通过

### 综合测试
- **Comprehensive_Test**: ✅ 255/255 通过
  - 3 后端 × 9 数据类型 × 9 操作 = 243 测试用例
  - 额外测试：部分跳过情况

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

### 标量参数示例（SCALE 操作）
```c
ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = alpha * a[i];
    }
);

// FLOAT32
float alpha = 2.0f;
ace_kernel_invoke(dev, _ace_get_vec_scale(), ACE_DTYPE_FLOAT32, N,
                  (void*[]){&n, &alpha, buf_a, buf_c},
                  (int[]){sizeof(int), sizeof(float), 0, 0}, 4);

// FLOAT16
uint16_t alpha = float_to_float16(2.0f);
ace_kernel_invoke(dev, _ace_get_vec_scale(), ACE_DTYPE_FLOAT16, N,
                  (void*[]){&n, &alpha, buf_a, buf_c},
                  (int[]){sizeof(int), sizeof(uint16_t), 0, 0}, 4);
```

## 更新日志

### v1.0.0 (最新)
- ✅ 完成三后端表驱动架构重构
- ✅ 修复 Vulkan 后端 FLOAT16/BF16/INT64/FLOAT64 支持
- ✅ 修复 Vulkan push constants 对齐问题
- ✅ 修复 OpenCL `__global` 限定符位置
- ✅ 修复 OpenCL 标量参数传递
- ✅ 所有测试通过 (9/9)
