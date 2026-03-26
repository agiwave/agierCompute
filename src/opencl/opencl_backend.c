/**
 * @file opencl_backend.c
 * @brief OpenCL backend using official OpenCL SDK
 */
#include "ace.h"
#include "../ace_backend_api.h"

#ifdef OPENCL_AVAILABLE

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Internal structures
 * ============================================================================ */

/* 内核缓存：每个设备一个哈希表 */
#define KERNEL_CACHE_SIZE 256

typedef struct ocl_kernel_s {
    int id;                /* 内核 ID */
    cl_kernel kernel;
    cl_program program;
    char* name;
    struct ocl_kernel_s* next;  /* 用于哈希表链式处理 */
} ocl_kernel_t;

typedef struct {
    ocl_kernel_t* buckets[KERNEL_CACHE_SIZE];
} ocl_kernel_cache_t;

typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char name[256];
    size_t total_mem;
    int compute_units;
    int max_threads;
    ocl_kernel_cache_t kernel_cache;  /* 内核缓存 */
} ocl_device_t;

typedef struct {
    cl_mem mem;
    size_t size;
} ocl_buffer_t;

static cl_platform_id g_platform;

/* ============================================================================
 * ACE -> OpenCL translation
 * ============================================================================ */

static char* translate_to_opencl(const char* name, const char* src, const char* type_name) {
    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;

    /* 添加 OpenCL 特殊类型定义 */
    const char* header = NULL;
    if (strcmp(type_name, "half") == 0) {
        header = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }

    char* p;
    while ((p = strstr(code, "T")) != NULL) {
        int is_type = 1;
        if (p > code) {
            char prev = p[-1];
            if ((prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z') ||
                (prev >= '0' && prev <= '9') || prev == '_') is_type = 0;
        }
        if (p[1]) {
            char next = p[1];
            if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') ||
                (next >= '0' && next <= '9') || next == '_') is_type = 0;
        }

        if (is_type) {
            size_t type_len = strlen(type_name);
            char* new_code = malloc(strlen(code) + type_len + 1);
            if (!new_code) { free(code); return NULL; }

            *p = '\0';
            strcpy(new_code, code);
            strcat(new_code, type_name);
            strcat(new_code, p + 1);
            free(code);
            code = new_code;
        } else {
            p++;
        }
    }

    /* 添加 __global 到指针参数 */
    char* with_global = NULL;
    char* ptr = strstr(code, "(");
    if (ptr) {
        char* end = strchr(ptr, ')');
        if (end) {
            /* 在指针类型前添加 __global */
            size_t prefix_len = ptr - code + 1;  /* 包括 ( */
            size_t params_len = end - ptr - 1;
            size_t suffix_len = strlen(end);
            
            /* 分配足够的空间 */
            with_global = malloc(strlen(code) + params_len + 50);
            if (!with_global) { free(code); return NULL; }
            
            /* 复制前缀 */
            memcpy(with_global, code, prefix_len);
            char* dst = with_global + prefix_len;
            
            /* 处理参数，添加 __global */
            char* param_start = ptr + 1;
            while (param_start < end) {
                /* 跳过空格 */
                while (param_start < end && (*param_start == ' ' || *param_start == '\t')) param_start++;
                
                /* 检查是否有 * */
                char* star = strchr(param_start, '*');
                if (star && star < end) {
                    /* 在 * 前添加 __global */
                    memcpy(dst, "__global ", 9);
                    dst += 9;
                    
                    /* 复制从 param_start 到 star 的内容 */
                    memcpy(dst, param_start, star - param_start);
                    dst += star - param_start;
                    
                    /* 复制 * 和后面的内容 */
                    *dst++ = '*';
                    param_start = star + 1;
                    
                    /* 跳过空格 */
                    while (param_start < end && (*param_start == ' ' || *param_start == '\t')) param_start++;
                    
                    /* 复制参数名 */
                    char* comma = strchr(param_start, ',');
                    if (!comma || comma > end) comma = end;
                    memcpy(dst, param_start, comma - param_start);
                    dst += comma - param_start;
                    param_start = comma;
                    
                    /* 复制逗号 */
                    if (param_start < end && *param_start == ',') {
                        *dst++ = ',';
                        param_start++;
                    }
                } else {
                    /* 没有 *，直接复制 */
                    char* comma = strchr(param_start, ',');
                    if (!comma || comma > end) comma = end;
                    memcpy(dst, param_start, comma - param_start);
                    dst += comma - param_start;
                    param_start = comma;
                    
                    if (param_start < end && *param_start == ',') {
                        *dst++ = ',';
                        param_start++;
                    }
                }
            }
            
            /* 复制后缀 */
            strcpy(dst, end);
            
            free(code);
            code = with_global;
        }
    }

    const char* params_start = strchr(code, '(');
    const char* params_end = strchr(code, ')');
    const char* body_start = strchr(code, '{');
    const char* body_end = strrchr(code, '}');

    if (!params_start || !params_end || !body_start || !body_end) {
        free(code);
        char* out = (char*)malloc(256);
        snprintf(out, 256, "__kernel void %s() { int GID = get_global_id(0); }\n", name);
        return out;
    }

    size_t params_len = params_end - params_start + 1;
    char* params = (char*)malloc(params_len + 1);
    strncpy(params, params_start, params_len);
    params[params_len] = '\0';

    size_t body_len = body_end - body_start - 1;

    size_t total_len = strlen(name) + params_len + body_len + 512 + (header ? strlen(header) : 0);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "__kernel void %s%s\n"
        "{\n"
        "    int GID = get_global_id(0);\n"
        "    int LID = get_local_id(0);\n"
        "    int BSIZE = get_local_size(0);\n"
        "    %.*s\n"
        "}\n",
        header ? header : "",
        name, params,
        (int)body_len, body_start + 1
    );

    free(params);
    free(code);
    return out;
}

/* ============================================================================
 * Backend operations
 * ============================================================================ */

static ace_error_t ocl_init(ace_backend_info_t* info) {
    cl_uint num_platforms;
    if (clGetPlatformIDs(1, &g_platform, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        printf("[OpenCL] No platforms found\n");
        return ACE_ERROR_BACKEND;
    }
    
    char name[128];
    clGetPlatformInfo(g_platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
    printf("[OpenCL] Backend initialized (platform: %s)\n", name);
    return ACE_OK;
}

static void ocl_shutdown(ace_backend_info_t* info) {
    /* OpenCL cleanup is automatic */
}

static ace_error_t ocl_device_count(int* count) {
    cl_uint num = 0;
    cl_int err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num);
        if (err != CL_SUCCESS) num = 0;
    }
    *count = num;
    return ACE_OK;
}

static ace_error_t ocl_device_get(int idx, void** dev) {
    cl_device_id devices[16];
    cl_uint num;
    
    cl_int err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 16, devices, &num);
    if (err != CL_SUCCESS || idx >= (int)num) {
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_CPU, 16, devices, &num);
        if (err != CL_SUCCESS || idx >= (int)num) {
            return ACE_ERROR_DEVICE;
        }
    }
    
    ocl_device_t* d = (ocl_device_t*)calloc(1, sizeof(*d));
    d->device = devices[idx];
    
    clGetDeviceInfo(d->device, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);
    
    cl_ulong mem;
    clGetDeviceInfo(d->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
    d->total_mem = mem;
    
    cl_uint cu;
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    d->compute_units = cu;
    
    size_t wg;
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
    d->max_threads = (int)wg;
    
    d->context = clCreateContext(NULL, 1, &d->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(d);
        return ACE_ERROR_DEVICE;
    }
    
    d->queue = clCreateCommandQueue(d->context, d->device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(d->context);
        free(d);
        return ACE_ERROR_DEVICE;
    }

    /* 初始化内核缓存 */
    memset(&d->kernel_cache, 0, sizeof(d->kernel_cache));

    *dev = d;
    return ACE_OK;
}

static void ocl_device_release(void* dev) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (d) {
        /* 释放内核缓存 */
        for (int i = 0; i < KERNEL_CACHE_SIZE; i++) {
            ocl_kernel_t* k = d->kernel_cache.buckets[i];
            while (k) {
                ocl_kernel_t* next = k->next;
                if (k->kernel) clReleaseKernel(k->kernel);
                if (k->program) clReleaseProgram(k->program);
                free(k->name);
                free(k);
                k = next;
            }
        }
        if (d->queue) clReleaseCommandQueue(d->queue);
        if (d->context) clReleaseContext(d->context);
        free(d);
    }
}

static ace_error_t ocl_device_props(void* dev, void* props) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;
    
    p->type = ACE_BACKEND_DEVICE_OPENCL;
    strncpy(p->name, d->name, sizeof(p->name) - 1);
    strcpy(p->vendor, "OpenCL");
    p->total_memory = d->total_mem;
    p->max_threads = d->max_threads;
    p->compute_units = d->compute_units;
    return ACE_OK;
}

static ace_error_t ocl_mem_alloc(void* dev, size_t size, void** ptr) {
    ocl_device_t* d = (ocl_device_t*)dev;
    
    cl_int err;
    cl_mem mem = clCreateBuffer(d->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) return ACE_ERROR_MEM;
    
    ocl_buffer_t* buf = (ocl_buffer_t*)calloc(1, sizeof(*buf));
    buf->mem = mem;
    buf->size = size;
    *ptr = buf;
    return ACE_OK;
}

static void ocl_mem_free(void* dev, void* ptr) {
    ocl_buffer_t* buf = (ocl_buffer_t*)ptr;
    if (buf) {
        clReleaseMemObject(buf->mem);
        free(buf);
    }
}

static ace_error_t ocl_mem_write(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)dst;
    cl_int err = clEnqueueWriteBuffer(d->queue, buf->mem, CL_TRUE, 0, size, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t ocl_mem_read(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)src;
    cl_int err = clEnqueueReadBuffer(d->queue, buf->mem, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t ocl_finish(void* dev) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_DEVICE;
    return (clFinish(d->queue) == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

static ace_error_t ocl_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                      ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_LAUNCH;

    /* 查找缓存的内核 */
    /* 内核 ID 规则：core_id * 16 + dtype，确保不同数据类型有不同缓存 */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    int bucket = kernel_id % KERNEL_CACHE_SIZE;
    ocl_kernel_t* cached = d->kernel_cache.buckets[bucket];
    while (cached) {
        if (cached->id == kernel_id) {
            break;  /* 找到缓存 */
        }
        cached = cached->next;
    }

    /* 如果未缓存，编译内核 */
    if (!cached) {
        /* 使用 kernel_def 中的数据类型 */
        const char* type_name = "float";
        switch ((ace_dtype_t)kernel_def->dtype) {
            case ACE_DTYPE_FLOAT32:  type_name = "float"; break;
            case ACE_DTYPE_FLOAT64:  type_name = "double"; break;
            case ACE_DTYPE_INT32:    type_name = "int"; break;
            case ACE_DTYPE_INT64:    type_name = "long"; break;
            case ACE_DTYPE_FLOAT16:  type_name = "half"; break;
            case ACE_DTYPE_BFLOAT16: type_name = "bfloat16"; break;
            case ACE_DTYPE_INT8:     type_name = "char"; break;
            case ACE_DTYPE_UINT8:    type_name = "uchar"; break;
            case ACE_DTYPE_INT16:    type_name = "short"; break;
            default: type_name = "float"; break;
        }

        char* translated = translate_to_opencl(kernel_def->name, kernel_def->src, type_name);

        cl_int err;
        const char* srcs[1] = { translated };
        size_t lens[1] = { strlen(translated) };

        cl_program prog = clCreateProgramWithSource(d->context, 1, srcs, lens, &err);
        if (err != CL_SUCCESS) {
            free(translated);
            return ACE_ERROR_COMPILE;
        }

        err = clBuildProgram(prog, 1, &d->device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(prog, d->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = (char*)malloc(log_size);
            clGetProgramBuildInfo(prog, d->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("[OpenCL] Compile error for %s (%s):\n%s\n", kernel_def->name, type_name, log);
            free(log);
            clReleaseProgram(prog);
            free(translated);
            return ACE_ERROR_COMPILE;
        }

        cl_kernel krn = clCreateKernel(prog, kernel_def->name, &err);
        free(translated);

        if (err != CL_SUCCESS) {
            clReleaseProgram(prog);
            return ACE_ERROR_COMPILE;
        }

        /* 创建缓存条目 */
        ocl_kernel_t* k = (ocl_kernel_t*)calloc(1, sizeof(*k));
        k->id = kernel_id;
        k->name = strdup(kernel_def->name);
        k->kernel = krn;
        k->program = prog;

        /* 添加到缓存 */
        k->next = d->kernel_cache.buckets[bucket];
        d->kernel_cache.buckets[bucket] = k;
        cached = k;
    }

    /* 设置参数 */
    for (int i = 0; i < n; i++) {
        if (sizes[i] == ACE_ARG_BUFFER) {
            ocl_buffer_t* buf = (ocl_buffer_t*)args[i];
            if (!buf || !buf->mem) {
                return ACE_ERROR_LAUNCH;
            }
            clSetKernelArg(cached->kernel, i, sizeof(cl_mem), &buf->mem);
        } else {
            clSetKernelArg(cached->kernel, i, sizeof(int), args[i]);
        }
    }

    size_t global[3] = { cfg->grid[0] * cfg->block[0], cfg->grid[1] * cfg->block[1], cfg->grid[2] * cfg->block[2] };
    size_t local[3] = { cfg->block[0], cfg->block[1], cfg->block[2] };

    cl_int err = clEnqueueNDRangeKernel(d->queue, cached->kernel, 3, NULL, global, local, 0, NULL, NULL);

    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

/* Backend registration */
static ace_backend_ops_t ocl_ops = {
    .init = ocl_init,
    .shutdown = ocl_shutdown,
    .device_count = ocl_device_count,
    .device_get = ocl_device_get,
    .device_release = ocl_device_release,
    .device_props = ocl_device_props,
    .mem_alloc = ocl_mem_alloc,
    .mem_free = ocl_mem_free,
    .mem_write = ocl_mem_write,
    .mem_read = ocl_mem_read,
    .finish = ocl_finish,
    .kernel_launch = ocl_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_OPENCL, "OpenCL", &ocl_ops)

#else

/* OpenCL SDK not available - provide stub */
static ace_backend_ops_t ocl_ops = {0};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_OPENCL, "OpenCL (unavailable)", &ocl_ops)

#endif /* OPENCL_AVAILABLE */