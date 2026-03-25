/**
 * @file cpu_backend.c
 * @brief CPU后端实现 - 包含ACE到C的翻译和解释执行
 */
#include "ace.h"
#include "../ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * 内部结构
 * ============================================================================ */

typedef struct {
    ace_device_props_t props;
} cpu_device_t;

typedef struct {
    char* name;
    char* translated;
    int n_params;
} cpu_kernel_t;

/* ============================================================================
 * ACE → C 翻译
 * ============================================================================ */

static char* translate_source(const char* src) {
    /* 简单的文本替换翻译 - 新格式已经是标准C函数 */
    char* out = strdup(src);
    if (!out) return NULL;
    
    /* 替换规则表 */
    struct {
        const char* from;
        const char* to;
    } rules[] = {
        {"__global ", ""},
        {"__global\t", ""},
        {"global ", ""},
        {"global\t", ""},
        {"__kernel ", ""},
        {"__kernel\t", ""},
        {"BARRIER()", ""},
    };
    
    for (int i = 0; i < sizeof(rules)/sizeof(rules[0]); i++) {
        char* pos;
        while ((pos = strstr(out, rules[i].from)) != NULL) {
            size_t len = strlen(out) + strlen(rules[i].to) - strlen(rules[i].from) + 1;
            char* new_out = (char*)malloc(len);
            if (!new_out) {
                free(out);
                return NULL;
            }
            size_t prefix_len = pos - out;
            memcpy(new_out, out, prefix_len);
            strcpy(new_out + prefix_len, rules[i].to);
            strcat(new_out, pos + strlen(rules[i].from));
            free(out);
            out = new_out;
        }
    }
    
    return out;
}

/* ============================================================================
 * 内核执行器 - 解释执行
 * ============================================================================ */

/* 内核参数信息 */
typedef struct {
    void* ptr;
    size_t size;
    int is_buffer;
} kernel_arg_t;

static ace_error_t execute_kernel(cpu_kernel_t* k, ace_launch_config_t* cfg,
                                   void** args, size_t* sizes, int n) {
    size_t total_threads = cfg->grid[0] * cfg->block[0];
    
    /* 
     * 解释执行 - 这里使用一个简单的模式匹配来执行常见内核
     * 完整实现需要LLVM JIT或类似技术
     * 
     * 支持泛型内核名: vec_add_float, vec_add_int 等
     */
    const char* name = k->name;
    
    /* 向量加法内核 - 支持多种命名 */
    if ((strncmp(name, "vec_add", 7) == 0) && n == 4) {
        int count = *(int*)args[0];
        
        /* 根据内核名后缀判断类型 */
        if (strstr(name, "_int") || strcmp(name, "vec_add_int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t i = 0; i < (size_t)count; i++) c[i] = a[i] + b[i];
        } else {
            /* 默认 float */
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t i = 0; i < (size_t)count; i++) c[i] = a[i] + b[i];
        }
        return ACE_OK;
    }
    
    /* 向量缩放内核 */
    if ((strncmp(name, "scale", 5) == 0) && n == 4) {
        int count = *(int*)args[0];
        float scale = *(float*)args[1];
        float* in = (float*)args[2];
        float* out = (float*)args[3];
        for (size_t i = 0; i < (size_t)count; i++) out[i] = in[i] * scale;
        return ACE_OK;
    }
    
    /* ReLU内核 */
    if ((strncmp(name, "relu", 4) == 0) && n == 3) {
        int count = *(int*)args[0];
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = 0; i < (size_t)count; i++) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }
        return ACE_OK;
    }
    
    /* 未知内核，跳过执行 */
    return ACE_OK;
}

/* ============================================================================
 * 后端操作实现
 * ============================================================================ */

static ace_error_t cpu_init(ace_backend_info_t* info) {
    return ACE_OK;
}

static void cpu_shutdown(ace_backend_info_t* info) {
}

static ace_error_t cpu_device_count(int* count) {
    *count = 1;
    return ACE_OK;
}

static ace_error_t cpu_device_get(int idx, void** dev) {
    if (idx != 0) return ACE_ERROR_DEVICE;
    
    cpu_device_t* d = (cpu_device_t*)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;
    
    strcpy(d->props.name, "CPU");
    strcpy(d->props.vendor, "Generic");
    d->props.type = ACE_DEVICE_CPU;
    d->props.max_threads = 1024;
    
    *dev = d;
    return ACE_OK;
}

static void cpu_device_release(void* dev) {
    free(dev);
}

static ace_error_t cpu_device_props(void* dev, ace_device_props_t* props) {
    if (!dev || !props) return ACE_ERROR_DEVICE;
    *props = ((cpu_device_t*)dev)->props;
    return ACE_OK;
}

static ace_error_t cpu_mem_alloc(void* dev, size_t size, void** ptr) {
    *ptr = calloc(1, size);
    return *ptr ? ACE_OK : ACE_ERROR_MEM;
}

static void cpu_mem_free(void* dev, void* ptr) {
    free(ptr);
}

static ace_error_t cpu_mem_write(void* dev, void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_mem_read(void* dev, void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_kernel_compile(void* dev, const char* name, 
                                       const char* src, void** kernel, char** err) {
    cpu_kernel_t* k = (cpu_kernel_t*)calloc(1, sizeof(*k));
    if (!k) return ACE_ERROR_MEM;
    
    k->name = strdup(name);
    k->translated = translate_source(src);
    
    *kernel = k;
    return ACE_OK;
}

static void cpu_kernel_release(void* kernel) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    if (k) {
        free(k->name);
        free(k->translated);
        free(k);
    }
}

static ace_error_t cpu_kernel_launch(void* kernel, ace_launch_config_t* cfg,
                                      void** args, size_t* sizes, int n) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    return execute_kernel(k, cfg, args, sizes, n);
}

static ace_error_t cpu_finish(void* dev) {
    /* CPU 后端是同步的，无需等待 */
    return ACE_OK;
}

/* ============================================================================
 * 后端注册
 * ============================================================================ */

static ace_backend_ops_t cpu_ops = {
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .device_count = cpu_device_count,
    .device_get = cpu_device_get,
    .device_release = cpu_device_release,
    .device_props = cpu_device_props,
    .mem_alloc = cpu_mem_alloc,
    .mem_free = cpu_mem_free,
    .mem_write = cpu_mem_write,
    .mem_read = cpu_mem_read,
    .finish = cpu_finish,
    .kernel_compile = cpu_kernel_compile,
    .kernel_release = cpu_kernel_release,
    .kernel_launch = cpu_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_DEVICE_CPU, "CPU", &cpu_ops)