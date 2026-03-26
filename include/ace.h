/**
 * @file ace.h
 * @brief AgierCompute - 跨平台GPU计算框架
 * 
 * 简洁的API设计，类似SYCL风格
 * 
 * 示例：
 *   // 1. 定义内核
 *   ACE_KERNEL(vec_add,
 *       void vec_add(int n, T* a, T* b, T* c) {
 *           int i = GID;
 *           if (i < n) c[i] = a[i] + b[i];
 *       }
 *   );
 *   
 *   // 2. 获取设备
 *   ace_device_t dev;
 *   ace_device_get(ACE_DEVICE_CPU, 0, &dev);
 *   
 *   // 3. 分配内存
 *   ace_buffer_t a, b, c;
 *   ace_buffer_alloc(dev, N * sizeof(float), &a);
 *   ace_buffer_alloc(dev, N * sizeof(float), &b);
 *   ace_buffer_alloc(dev, N * sizeof(float), &c);
 *   
 *   // 4. 写入数据
 *   ace_buffer_write(a, h_a, N * sizeof(float));
 *   ace_buffer_write(b, h_b, N * sizeof(float));
 *   
 *   // 5. 执行内核（自动异步）
 *   int n = N;
 *   void* args[] = {&n, a, b, c};
 *   int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
 *   ace_kernel_invoke(dev, k_vec_add, ACE_DTYPE_FLOAT32, N, args, types, 4);
 *   
 *   // 6. 读取结果（自动同步）
 *   ace_buffer_read(c, h_c, N * sizeof(float));
 *   
 *   // 7. 清理
 *   ace_buffer_free(a); ace_buffer_free(b); ace_buffer_free(c);
 *   ace_device_release(dev);
 */
#ifndef ACE_H
#define ACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================================
 * 版本信息
 * ============================================================================ */

#define ACE_VERSION_MAJOR 1
#define ACE_VERSION_MINOR 0
#define ACE_VERSION_PATCH 0
#define ACE_VERSION "1.0.0"

/* ============================================================================
 * 平台导出宏
 * ============================================================================ */

#ifdef _WIN32
    #ifdef ACE_CORE_EXPORTS
        #define ACE_API __declspec(dllexport)
    #else
        #define ACE_API __declspec(dllimport)
    #endif
#else
    #define ACE_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * 错误码
 * ============================================================================ */

typedef int ace_error_t;

#define ACE_OK              0
#define ACE_ERROR          -1
#define ACE_ERROR_MEM      -2
#define ACE_ERROR_DEVICE   -3
#define ACE_ERROR_COMPILE  -4
#define ACE_ERROR_LAUNCH   -5
#define ACE_ERROR_IO       -6
#define ACE_ERROR_BACKEND  -7
#define ACE_ERROR_NOT_FOUND -8
#define ACE_ERROR_INVALID   -9

/* ============================================================================
 * 设备类型
 * ============================================================================ */

typedef enum {
    ACE_DEVICE_CPU    = 0,
    ACE_DEVICE_CUDA   = 1,
    ACE_DEVICE_OPENCL = 2,
    ACE_DEVICE_VULKAN = 3,
    ACE_DEVICE_METAL  = 4,
    ACE_DEVICE_ALL    = 5,  /* 遍历所有类型设备 */
    ACE_DEVICE_COUNT  = 6,
} ace_device_type_t;

/* ============================================================================
 * 数据类型
 * ============================================================================ */

typedef enum {
    ACE_DTYPE_FLOAT32 = 0,
    ACE_DTYPE_FLOAT64 = 1,
    ACE_DTYPE_INT32   = 2,
    ACE_DTYPE_INT64   = 3,
    /* AI 重要数据类型 */
    ACE_DTYPE_FLOAT16 = 4,  /* FP16 - 半精度浮点 */
    ACE_DTYPE_BFLOAT16 = 5, /* BF16 - Brain 浮点 */
    ACE_DTYPE_INT8    = 6,  /* INT8 - 8 位整数 */
    ACE_DTYPE_UINT8   = 7,  /* UINT8 - 8 位无符号整数 */
    ACE_DTYPE_INT16   = 8,  /* INT16 - 16 位整数 */
    ACE_DTYPE_BOOL    = 9,  /* BOOL - 布尔值 */
} ace_dtype_t;

/* 数据类型大小辅助函数 */
static inline size_t ace_dtype_size(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return 4;
        case ACE_DTYPE_FLOAT64:  return 8;
        case ACE_DTYPE_INT32:    return 4;
        case ACE_DTYPE_INT64:    return 8;
        case ACE_DTYPE_FLOAT16:  return 2;
        case ACE_DTYPE_BFLOAT16: return 2;
        case ACE_DTYPE_INT8:     return 1;
        case ACE_DTYPE_UINT8:    return 1;
        case ACE_DTYPE_INT16:    return 2;
        case ACE_DTYPE_BOOL:     return 1;
        default:                 return 4;
    }
}

/* 获取数据类型名称 */
static inline const char* ace_dtype_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float32";
        case ACE_DTYPE_FLOAT64:  return "float64";
        case ACE_DTYPE_INT32:    return "int32";
        case ACE_DTYPE_INT64:    return "int64";
        case ACE_DTYPE_FLOAT16:  return "float16";
        case ACE_DTYPE_BFLOAT16: return "bfloat16";
        case ACE_DTYPE_INT8:     return "int8";
        case ACE_DTYPE_UINT8:    return "uint8";
        case ACE_DTYPE_INT16:    return "int16";
        case ACE_DTYPE_BOOL:     return "bool";
        default:                 return "float32";
    }
}

/* ============================================================================
 * AI 数据类型转换辅助函数
 * ============================================================================ */

/* FP16 (float16) 转换辅助函数 */
typedef uint16_t ace_float16_t;

static inline ace_float16_t float_to_float16(float f) {
    union { float f; uint32_t u; } u = {f};
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exp = ((u.u >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (u.u >> 13) & 0x3ff;

    if (exp <= 0) return (ace_float16_t)sign;
    if (exp >= 31) return (ace_float16_t)(sign | 0x7c00);
    return (ace_float16_t)(sign | (exp << 10) | frac);
}

static inline float float16_to_float(ace_float16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = ((h & 0x7c00) >> 10);
    uint32_t frac = (h & 0x03ff) << 13;

    if (exp == 0) exp = 0;
    else if (exp == 31) exp = 255;
    else exp += 127 - 15;

    union { float f; uint32_t u; } u;
    u.u = sign | (exp << 23) | frac;
    return u.f;
}

/* BF16 (bfloat16) 转换辅助函数 */
typedef uint16_t ace_bfloat16_t;

static inline ace_bfloat16_t float_to_bfloat16(float f) {
    union { float f; uint32_t u; } u = {f};
    return (ace_bfloat16_t)(u.u >> 16);
}

static inline float bfloat16_to_float(ace_bfloat16_t h) {
    union { float f; uint32_t u; } u;
    u.u = (uint32_t)h << 16;
    return u.f;
}

/* ============================================================================
 * 设备属性
 * ============================================================================ */

typedef struct {
    ace_device_type_t type;
    char name[256];
    char vendor[128];
    size_t total_memory;
    size_t max_threads;
    int compute_units;
} ace_device_props_t;

/* ============================================================================
 * 不透明句柄
 * ============================================================================ */

typedef struct ace_device_* ace_device_t;
typedef struct ace_buffer_* ace_buffer_t;
typedef void* ace_kernel_t;

/* ============================================================================
 * 参数传递设计 - 使用 sizeof() 表示参数大小
 * ============================================================================ */

/* 参数大小规则：
 * - 正数 (>0)：表示标量参数的字节数，如 sizeof(int)、sizeof(float) 等
 * - 0 或负数：表示缓冲区 (ace_buffer_t)
 */
#define ACE_ARG_BUFFER  0   /* 缓冲区 */

/* ============================================================================
 * 3D调度配置
 * ============================================================================ */

struct ace_launch_config_ {
    size_t grid[3];     /* 工作组数量 */
    size_t block[3];    /* 每个工作组的线程数 */
    size_t shared_mem;  /* 动态共享内存大小（高级功能，默认0） */
};

typedef struct ace_launch_config_ ace_launch_config_t;

/* 调度配置辅助函数 */
static inline ace_launch_config_t ace_launch_1d(size_t n, size_t block) {
    ace_launch_config_t cfg = { 
        .grid = {(n + block - 1) / block, 1, 1}, 
        .block = {block, 1, 1},
        .shared_mem = 0
    };
    return cfg;
}

static inline ace_launch_config_t ace_launch_2d(size_t nx, size_t ny, size_t bx, size_t by) {
    ace_launch_config_t cfg = { 
        .grid = {(nx + bx - 1) / bx, (ny + by - 1) / by, 1},
        .block = {bx, by, 1},
        .shared_mem = 0
    };
    return cfg;
}

static inline ace_launch_config_t ace_launch_3d(size_t nx, size_t ny, size_t nz,
                                                  size_t bx, size_t by, size_t bz) {
    ace_launch_config_t cfg = { 
        .grid = {(nx + bx - 1) / bx, (ny + by - 1) / by, (nz + bz - 1) / bz},
        .block = {bx, by, bz},
        .shared_mem = 0
    };
    return cfg;
}

/* 简化宏 */
#define ACE_1D(n) ace_launch_1d(n, 256)
#define ACE_1D_BLOCK(n, b) ace_launch_1d(n, b)

/* ============================================================================
 * 内核定义宏
 * ============================================================================ */

#define ACE_KERNEL(name, code) \
    static ace_kernel_t k_##name = NULL; \
    static const char* _ace_src_##name = #code; \
    static ace_kernel_t _ace_get_##name(void) { \
        if (!k_##name) k_##name = ace_kernel_register(#name, _ace_src_##name); \
        return k_##name; \
    }

/* ============================================================================
 * 简化宏 - 推荐使用
 * ============================================================================ */

/* 错误检查宏 - 用于返回 ace_error_t 的函数 */
#define ACE_CHECK(call) do { \
    ace_error_t _err = (call); \
    if (_err != ACE_OK) { \
        fprintf(stderr, "ACE error at %s:%d: %s\n", __FILE__, __LINE__, ace_error_string(_err)); \
        return _err; \
    } \
} while(0)

/* 错误检查宏 - 用于 void 函数 */
#define ACE_CHECK_VOID(call) do { \
    ace_error_t _err = (call); \
    if (_err != ACE_OK) { \
        fprintf(stderr, "ACE error at %s:%d: %s\n", __FILE__, __LINE__, ace_error_string(_err)); \
        return; \
    } \
} while(0)

#ifndef __KARGS__

    /* ACE_INVOKE: 标准宏 - 自动推断参数大小
    * 用法：ACE_INVOKE(dev, vec_add, FLOAT32, N, &n, buf_a, buf_b, buf_c);
    *      ACE_INVOKE(dev, scale, FLOAT32, N, &n, &alpha, buf_in, buf_out);
    * 说明：自动推断参数大小 - 指针视为 buffer，其他根据 sizeof 推断大小
    */
    #define ACE_INVOKE(dev, kernel_name, dtype, n, ...) \
        do { \
            void* _args[] = {__VA_ARGS__}; \
            int _nargs = sizeof(_args) / sizeof(_args[0]); \
            int _sizes[16] = {0}; \
            for (int _i = 0; _i < _nargs && _i < 16; _i++) { \
                _sizes[_i] = (_i == 0) ? sizeof(int) : ACE_ARG_BUFFER; \
            } \
            ace_kernel_invoke(dev, _ace_get_##kernel_name(), dtype, n, _args, _sizes, _nargs); \
        } while(0)

#else//

    struct KInvoker {
        static const int MAX_INVOKE_ARGS = 20;
        enum EInvokeArgType{
            EBuffer,
            EDataPointer,
            EData  
        };
        int nArgs;
        int pArgSizes[MAX_INVOKE_ARGS];
        EInvokeArgType pArgType[MAX_INVOKE_ARGS];
        union{
            void* ptr;
            uint8_t data[8];
        }pArgs[MAX_INVOKE_ARGS];
        template<typename T, typename... Args>
        KInvoker(const T& first, Args... rest) : KInvoker(rest...){
            assert(nArgs<MAX_INVOKE_ARGS);
            assert(!(std::is_pointer_v<T>));
            if(sizeof(T) <= 8) {
                pArgType[nArgs] = EData;
                *(T*)pArgs[nArgs].data = first;
            }else{
                pArgType[nArgs] = EDataPointer;
                pArgs[nArgs].ptr = (void*)&first;
            }
            pArgSizes[nArgs++] = sizeof(T);
        }
        template<typename... Args>
        KInvoker(ace_buffer_t* first, Args... rest) : KInvoker(rest...){
            assert(nArgs<MAX_INVOKE_ARGS);
            pArgSizes[nArgs] = 0;
            pArgType[nArgs] = EBuffer;
            pArgs[nArgs++].ptr = first;
        }
        KInvoker() : nArgs(0){};

        ace_error_t invoke(ace_device_t device, ace_kernel_t kernel, ace_dtype_t dtype, size_t n) {
            void* _args[MAX_INVOKE_ARGS];
            for (int i = 0; i < nArgs; i++) {
                _args[i] = (pArgType[i] == EBuffer) ? pArgs[i].ptr : (void*)pArgs[i].data;
            }
            return ace_kernel_invoke(device, kernel, dtype, n, _args, _pArgSizes, nArgs);
        }
    };


    /* ACE_INVOKE: 标准宏 - 自动推断参数大小
    * 用法：ACE_INVOKE(dev, vec_add, FLOAT32, N, &n, buf_a, buf_b, buf_c);
    *      ACE_INVOKE(dev, scale, FLOAT32, N, &n, &alpha, buf_in, buf_out);
    * 说明：自动推断参数大小 - 指针视为 buffer，其他根据 sizeof 推断大小
    */
    #define ACE_INVOKE(dev, kernel_name, dtype, n, ...) \
        KInvoker(__VA_ARGS__).invoke(dev, _ace_get_##kernel_name(), dtype, n)

#endif//

#define LID        /* 局部线程ID */
#define BSIZE      /* 工作组大小 */
#define BARRIER()  /* 局部同步 */

/* ============================================================================
 * 核心 API
 * ============================================================================ */

/* ----------------------------------------------------------------------------
 * 设备管理
 * ---------------------------------------------------------------------------- */

/* 获取指定类型的设备数量（type=ACE_DEVICE_ALL 表示所有类型） */
ACE_API ace_error_t ace_device_count(ace_device_type_t type, int* count);

/* 获取设备 */
ACE_API ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev);

/* 释放设备 */
ACE_API void ace_device_release(ace_device_t dev);

/* 获取设备属性 */
ACE_API ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props);

/* ----------------------------------------------------------------------------
 * 内存管理（框架自动池化）
 * ---------------------------------------------------------------------------- */

/* 分配设备内存 */
ACE_API ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);

/* 释放设备内存 */
ACE_API void ace_buffer_free(ace_buffer_t buf);

/* 写入数据到设备（异步） */
ACE_API ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);

/* 从设备读取数据（自动同步） */
ACE_API ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);

/* ----------------------------------------------------------------------------
 * 内核管理
 * ---------------------------------------------------------------------------- */

/* 注册内核 */
ACE_API ace_kernel_t ace_kernel_register(const char* name, const char* src);

/* 简化的内核调用 - 1D调度，自动异步 */
ACE_API ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, size_t n,
                                       void** args, int* types, int nargs);

/* 高级内核调用 - 支持自定义3D调度 */
ACE_API ace_error_t ace_kernel_launch(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, ace_launch_config_t* config,
                                       void** args, int* types, int nargs);

/* ----------------------------------------------------------------------------
 * 同步（可选，buffer_read会自动同步）
 * ---------------------------------------------------------------------------- */

/* 等待设备上所有操作完成 */
ACE_API ace_error_t ace_finish(ace_device_t dev);

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/* 获取错误描述 */
static inline const char* ace_error_string(ace_error_t err) {
    switch (err) {
        case ACE_OK:              return "OK";
        case ACE_ERROR:           return "General error";
        case ACE_ERROR_MEM:       return "Memory error";
        case ACE_ERROR_DEVICE:    return "Device error";
        case ACE_ERROR_COMPILE:   return "Compile error";
        case ACE_ERROR_LAUNCH:    return "Launch error";
        case ACE_ERROR_IO:        return "I/O error";
        case ACE_ERROR_BACKEND:   return "Backend error";
        case ACE_ERROR_NOT_FOUND: return "Not found";
        case ACE_ERROR_INVALID:   return "Invalid argument";
        default:                  return "Unknown error";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ACE_H */
