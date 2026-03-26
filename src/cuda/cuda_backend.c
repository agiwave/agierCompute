/**
 * @file cuda_backend.c
 * @brief CUDA backend using official CUDA SDK
 */
#include "ace.h"
#include "../ace_backend_api.h"

#ifdef CUDA_AVAILABLE

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NVRTC_AVAILABLE
#include <nvrtc.h>
#endif

/* ============================================================================
 * Internal structures
 * ============================================================================ */

/* 内核缓存：每个设备一个哈希表 */
#define KERNEL_CACHE_SIZE 256

typedef struct cuda_kernel_s {
    int id;                /* 内核 ID */
    CUmodule module;
    CUfunction func;
    char* name;
    struct cuda_kernel_s* next;  /* 用于哈希表链式处理 */
} cuda_kernel_t;

typedef struct {
    cuda_kernel_t* buckets[KERNEL_CACHE_SIZE];
} cuda_kernel_cache_t;

typedef struct {
    CUdevice device;
    CUcontext context;
    char name[256];
    size_t total_mem;
    int sm_count;
    int max_threads;
    int compute_major;
    int compute_minor;
    cuda_kernel_cache_t kernel_cache;  /* 内核缓存 */
} cuda_device_t;

typedef struct {
    CUdeviceptr ptr;
    size_t size;
} cuda_buffer_t;

/* ============================================================================
 * ACE -> CUDA translation
 * ============================================================================ */

static char* translate_to_cuda(const char* name, const char* src, const char* type_name) {
    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;
    
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
            size_t rest_len = strlen(p + 1);
            size_t type_len = strlen(type_name);
            char* new_code = malloc(strlen(code) + type_len);
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
    
    const char* params_start = strchr(code, '(');
    const char* params_end = strchr(code, ')');
    const char* body_start = strchr(code, '{');
    const char* body_end = strrchr(code, '}');

    if (!params_start || !params_end || !body_start || !body_end) {
        free(code);
        char* out = (char*)malloc(256);
        snprintf(out, 256, "extern \"C\" __global__ void %s() {}\n", name);
        return out;
    }

    size_t params_len = params_end - params_start + 1;
    char* params = (char*)malloc(params_len + 1);
    strncpy(params, params_start, params_len);
    params[params_len] = '\0';

    size_t body_len = body_end - body_start - 1;

    size_t total_len = strlen(name) + params_len + body_len + 512;
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "extern \"C\" __global__ void %s%s\n"
        "{\n"
        "    const int GID = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    const int LID = threadIdx.x;\n"
        "    const int BSIZE = blockDim.x;\n"
        "    %.*s\n"
        "}\n",
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

static ace_error_t cuda_init(ace_backend_info_t* info) {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        printf("[CUDA] cuInit failed: %d\n", err);
        return ACE_ERROR_BACKEND;
    }
    
#ifdef NVRTC_AVAILABLE
    int major, minor;
    nvrtcVersion(&major, &minor);
    printf("[CUDA] Backend initialized (NVRTC %d.%d)\n", major, minor);
#else
    printf("[CUDA] Backend initialized (no NVRTC)\n");
#endif
    
    return ACE_OK;
}

static void cuda_shutdown(ace_backend_info_t* info) {
    /* CUDA driver cleanup is automatic */
}

static ace_error_t cuda_device_count(int* count) {
    int c = 0;
    CUresult err = cuDeviceGetCount(&c);
    *count = (err == CUDA_SUCCESS) ? c : 0;
    return ACE_OK;
}

static ace_error_t cuda_device_get(int idx, void** dev) {
    cuda_device_t* d = (cuda_device_t*)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;
    
    if (cuDeviceGet(&d->device, idx) != CUDA_SUCCESS) {
        free(d);
        return ACE_ERROR_DEVICE;
    }
    
    cuDeviceGetName(d->name, sizeof(d->name), d->device);
    cuDeviceTotalMem(&d->total_mem, d->device);
    cuDeviceGetAttribute(&d->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d->device);
    cuDeviceGetAttribute(&d->max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, d->device);
    cuDeviceGetAttribute(&d->compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d->device);
    cuDeviceGetAttribute(&d->compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d->device);
    
    if (cuCtxCreate(&d->context, 0, d->device) != CUDA_SUCCESS) {
        free(d);
        return ACE_ERROR_DEVICE;
    }

    /* 初始化内核缓存 */
    memset(&d->kernel_cache, 0, sizeof(d->kernel_cache));

    *dev = d;
    return ACE_OK;
}

static void cuda_device_release(void* dev) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (d) {
        /* 释放内核缓存 */
        for (int i = 0; i < KERNEL_CACHE_SIZE; i++) {
            cuda_kernel_t* k = d->kernel_cache.buckets[i];
            while (k) {
                cuda_kernel_t* next = k->next;
                if (k->module) cuModuleUnload(k->module);
                free(k->name);
                free(k);
                k = next;
            }
        }
        if (d->context) cuCtxDestroy(d->context);
        free(d);
    }
}

static ace_error_t cuda_device_props(void* dev, void* props) {
    cuda_device_t* d = (cuda_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;
    
    p->type = ACE_BACKEND_DEVICE_CUDA;
    strncpy(p->name, d->name, sizeof(p->name) - 1);
    strcpy(p->vendor, "NVIDIA");
    p->total_memory = d->total_mem;
    p->max_threads = d->max_threads;
    p->compute_units = d->sm_count;
    return ACE_OK;
}

static ace_error_t cuda_mem_alloc(void* dev, size_t size, void** ptr) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;

    cuCtxSetCurrent(d->context);

    cuda_buffer_t* buf = (cuda_buffer_t*)calloc(1, sizeof(*buf));
    if (!buf) return ACE_ERROR_MEM;

    if (cuMemAlloc(&buf->ptr, size) != CUDA_SUCCESS) {
        free(buf);
        return ACE_ERROR_MEM;
    }

    buf->size = size;
    *ptr = buf;
    return ACE_OK;
}

static void cuda_mem_free(void* dev, void* ptr) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)ptr;
    if (buf) {
        if (buf->ptr) cuMemFree(buf->ptr);
        free(buf);
    }
}

static ace_error_t cuda_mem_write(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)dst;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyHtoD(buf->ptr, src, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t cuda_mem_read(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)src;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyDtoH(dst, buf->ptr, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t cuda_finish(void* dev) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;
    cuCtxSetCurrent(d->context);
    return (cuCtxSynchronize() == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#ifdef NVRTC_AVAILABLE
/* NVRTC available - kernel compilation is handled in cuda_kernel_launch */
#endif /* NVRTC_AVAILABLE */

static ace_error_t cuda_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                       ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;

    cuCtxSetCurrent(d->context);

    /* 查找缓存的内核 */
    /* 内核 ID 规则：core_id * 16 + dtype，确保不同数据类型有不同缓存 */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    int bucket = kernel_id % KERNEL_CACHE_SIZE;
    cuda_kernel_t* cached = d->kernel_cache.buckets[bucket];
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
            case ACE_DTYPE_BFLOAT16: type_name = "__nv_bfloat16"; break;
            case ACE_DTYPE_INT8:     type_name = "char"; break;
            case ACE_DTYPE_UINT8:    type_name = "unsigned char"; break;
            case ACE_DTYPE_INT16:    type_name = "short"; break;
            default: type_name = "float"; break;
        }

        char* cuda_src = translate_to_cuda(kernel_def->name, kernel_def->src, type_name);

#ifdef NVRTC_AVAILABLE
        nvrtcProgram prog;
        nvrtcResult res = nvrtcCreateProgram(&prog, cuda_src, kernel_def->name, 0, NULL, NULL);
        if (res != NVRTC_SUCCESS) {
            free(cuda_src);
            return ACE_ERROR_COMPILE;
        }

        char arch_opt[64];
        snprintf(arch_opt, sizeof(arch_opt), "-arch=compute_%d%d", d->compute_major, d->compute_minor);
        const char* opts[] = { arch_opt, "-default-device" };

        res = nvrtcCompileProgram(prog, 2, opts);
        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            free(cuda_src);
            return ACE_ERROR_COMPILE;
        }

        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);
        char* ptx = (char*)malloc(ptx_size);
        nvrtcGetPTX(prog, ptx);
        nvrtcDestroyProgram(&prog);
        free(cuda_src);

        cuda_kernel_t* k = (cuda_kernel_t*)calloc(1, sizeof(*k));
        if (!k) {
            free(ptx);
            return ACE_ERROR_MEM;
        }

        k->id = kernel_id;
        k->name = strdup(kernel_def->name);

        if (cuModuleLoadData(&k->module, ptx) != CUDA_SUCCESS) {
            free(ptx);
            free(k->name);
            free(k);
            return ACE_ERROR_COMPILE;
        }
        free(ptx);

        if (cuModuleGetFunction(&k->func, k->module, kernel_def->name) != CUDA_SUCCESS) {
            cuModuleUnload(k->module);
            free(k->name);
            free(k);
            return ACE_ERROR_COMPILE;
        }

        /* 添加到缓存 */
        k->next = d->kernel_cache.buckets[bucket];
        d->kernel_cache.buckets[bucket] = k;
        cached = k;
#else
        free(cuda_src);
        return ACE_ERROR_COMPILE;
#endif
    }

    /* 执行内核 */
    void* kernel_args[16];
    CUdeviceptr ptr_values[16];
    
    for (int i = 0; i < n && i < 16; i++) {
        if (sizes[i] == ACE_ARG_BUFFER) {
            cuda_buffer_t* buf = (cuda_buffer_t*)args[i];
            ptr_values[i] = buf->ptr;
            kernel_args[i] = &ptr_values[i];
        } else {
            kernel_args[i] = args[i];
        }
    }

    CUresult err = cuLaunchKernel(
        cached->func,
        (unsigned int)cfg->grid[0], (unsigned int)cfg->grid[1], (unsigned int)cfg->grid[2],
        (unsigned int)cfg->block[0], (unsigned int)cfg->block[1], (unsigned int)cfg->block[2],
        (unsigned int)cfg->shared_mem, NULL,
        kernel_args, NULL
    );

    if (err != CUDA_SUCCESS) {
        printf("[CUDA] cuLaunchKernel failed: %d\n", err);
        return ACE_ERROR_LAUNCH;
    }
    
    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        printf("[CUDA] cuCtxSynchronize failed: %d\n", err);
        return ACE_ERROR_LAUNCH;
    }
    
    return ACE_OK;
}

/* Backend registration */
static ace_backend_ops_t cuda_ops = {
    .init = cuda_init,
    .shutdown = cuda_shutdown,
    .device_count = cuda_device_count,
    .device_get = cuda_device_get,
    .device_release = cuda_device_release,
    .device_props = cuda_device_props,
    .mem_alloc = cuda_mem_alloc,
    .mem_free = cuda_mem_free,
    .mem_write = cuda_mem_write,
    .mem_read = cuda_mem_read,
    .finish = cuda_finish,
    .kernel_launch = cuda_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CUDA, "CUDA", &cuda_ops)

#else

/* CUDA SDK not available - provide stub */
static ace_backend_ops_t cuda_ops = {0};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CUDA, "CUDA (unavailable)", &cuda_ops)

#endif /* CUDA_AVAILABLE */
