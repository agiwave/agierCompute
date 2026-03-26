/**
 * @file ace_backend_api.h
 * @brief 后端插件API - 后端开发需要
 * 
 * 用户不需要包含此文件，只需包含 ace.h
 */
#ifndef ACE_BACKEND_API_H
#define ACE_BACKEND_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 后端导出宏
 * ============================================================================ */

#ifdef _WIN32
    #ifdef ACE_BACKEND_EXPORTS
        #define ACE_BACKEND_API __declspec(dllexport)
    #else
        #define ACE_BACKEND_API __declspec(dllimport)
    #endif
#else
    #define ACE_BACKEND_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * 类型定义（后端内部使用）
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

/* 设备类型 - 使用int以兼容枚举 */
typedef enum {
    ACE_BACKEND_DEVICE_CPU    = 0,
    ACE_BACKEND_DEVICE_CUDA   = 1,
    ACE_BACKEND_DEVICE_OPENCL = 2,
    ACE_BACKEND_DEVICE_VULKAN = 3,
    ACE_BACKEND_DEVICE_METAL  = 4,
} ace_backend_device_type_t;

/* 参数类型标记 */
#define ACE_ARG_BUFFER  0
#define ACE_ARG_VALUE   1

/* 启动配置 - 前向声明（完整定义在 ace.h 中）*/
typedef struct ace_launch_config_ ace_launch_config_t;

/* ============================================================================
 * 后端信息结构
 * ============================================================================ */

typedef struct {
    ace_backend_device_type_t type;
    const char* name;
    void* user_data;
} ace_backend_info_t;

/* ============================================================================
 * 内核定义结构
 * ============================================================================ */

typedef struct {
    int id;              /* 内核 ID（唯一标识） */
    const char* name;    /* 内核名称 */
    const char* src;     /* 内核源代码 */
    int dtype;           /* 数据类型 */
} ace_kernel_def_t;

/* ============================================================================
 * 后端操作函数表
 * ============================================================================ */

typedef struct {
    ace_error_t (*init)(ace_backend_info_t* info);
    void (*shutdown)(ace_backend_info_t* info);

    ace_error_t (*device_count)(int* count);
    ace_error_t (*device_get)(int idx, void** dev);
    void (*device_release)(void* dev);
    ace_error_t (*device_props)(void* dev, void* props);

    ace_error_t (*mem_alloc)(void* dev, size_t size, void** ptr);
    void (*mem_free)(void* dev, void* ptr);
    ace_error_t (*mem_write)(void* dev, void* dst, const void* src, size_t size);
    ace_error_t (*mem_read)(void* dev, void* dst, const void* src, size_t size);
    ace_error_t (*finish)(void* dev);  /* 同步等待 */
    
    /* 内核执行：后端负责编译和缓存 */
    ace_error_t (*kernel_launch)(void* dev, ace_kernel_def_t* kernel_def,
                                  ace_launch_config_t* cfg, void** args, size_t* sizes, int n);
} ace_backend_ops_t;

/* ============================================================================
 * 后端注册宏
 * ============================================================================ */

/* 后端必须导出的函数 */
typedef ace_backend_info_t* (*ace_get_backend_fn)(void);
typedef ace_backend_ops_t* (*ace_get_ops_fn)(void);

/* 后端定义宏 - 同时导出 info 和 ops */
#define ACE_DEFINE_BACKEND(type_, name_, ops_ptr_) \
    static ace_backend_info_t _g_backend_info; \
    static ace_backend_ops_t* _g_backend_ops_ptr = NULL; \
    static int _g_backend_inited = 0; \
    ACE_BACKEND_API ace_backend_info_t* ace_get_backend(void) { \
        if (!_g_backend_inited) { \
            _g_backend_info.type = type_; \
            _g_backend_info.name = name_; \
            _g_backend_info.user_data = NULL; \
            _g_backend_ops_ptr = ops_ptr_; \
            _g_backend_inited = 1; \
        } \
        return &_g_backend_info; \
    } \
    ACE_BACKEND_API ace_backend_ops_t* ace_get_backend_ops(void) { \
        return _g_backend_ops_ptr; \
    }

#ifdef __cplusplus
}
#endif

#endif /* ACE_BACKEND_API_H */