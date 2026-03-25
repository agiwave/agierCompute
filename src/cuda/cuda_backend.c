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

typedef struct {
    CUdevice device;
    CUcontext context;
    char name[256];
    size_t total_mem;
    int sm_count;
    int max_threads;
    int compute_major;
    int compute_minor;
} cuda_device_t;

typedef struct {
    void* ptr;
    size_t size;
    cuda_device_t* dev;
} cuda_buffer_t;

typedef struct {
    CUmodule module;
    CUfunction func;
    char* name;
    cuda_device_t* dev;
} cuda_kernel_t;

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
    
    *dev = d;
    return ACE_OK;
}

static void cuda_device_release(void* dev) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (d) {
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
    buf->dev = d;
    *ptr = buf;
    return ACE_OK;
}

static void cuda_mem_free(void* dev, void* ptr) {
    cuda_buffer_t* buf = (cuda_buffer_t*)ptr;
    if (buf) {
        if (buf->ptr) cuMemFree((CUdeviceptr)buf->ptr);
        free(buf);
    }
}

static ace_error_t cuda_mem_write(void* dev, void* dst, const void* src, size_t size) {
    cuda_buffer_t* buf = (cuda_buffer_t*)dst;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyHtoD((CUdeviceptr)buf->ptr, src, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t cuda_mem_read(void* dev, void* dst, const void* src, size_t size) {
    cuda_buffer_t* buf = (cuda_buffer_t*)src;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyDtoH(dst, (CUdeviceptr)buf->ptr, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t cuda_finish(void* dev) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;
    cuCtxSetCurrent(d->context);
    return (cuCtxSynchronize() == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#ifdef NVRTC_AVAILABLE

static ace_error_t cuda_kernel_compile(void* dev, const char* name, const char* src,
                                        void** kernel, char** err_msg) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;

    /* 确定数据类型 */
    const char* type_name = "float";
    const char* suffix = strrchr(name, '_');
    if (suffix) {
        suffix++;
        if (strcmp(suffix, "int") == 0 || strcmp(suffix, "int32") == 0) type_name = "int";
        else if (strcmp(suffix, "double") == 0 || strcmp(suffix, "float64") == 0) type_name = "double";
        else if (strcmp(suffix, "long") == 0 || strcmp(suffix, "int64") == 0) type_name = "long";
    }

    char* cuda_src = translate_to_cuda(name, src, type_name);
    
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, cuda_src, name, 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        free(cuda_src);
        if (err_msg) *err_msg = strdup("Failed to create NVRTC program");
        return ACE_ERROR_COMPILE;
    }
    
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "-arch=compute_%d%d", d->compute_major, d->compute_minor);
    const char* opts[] = { arch_opt, "-default-device" };
    
    res = nvrtcCompileProgram(prog, 2, opts);
    if (res != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (err_msg) {
            *err_msg = (char*)malloc(log_size + 1);
            nvrtcGetProgramLog(prog, *err_msg);
        }
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
    
    cuCtxSetCurrent(d->context);
    
    cuda_kernel_t* k = (cuda_kernel_t*)calloc(1, sizeof(*k));
    if (!k) {
        free(ptx);
        return ACE_ERROR_MEM;
    }
    
    k->name = strdup(name);
    k->dev = d;
    
    CUresult err = cuModuleLoadData(&k->module, ptx);
    free(ptx);
    
    if (err != CUDA_SUCCESS) {
        free(k->name);
        free(k);
        if (err_msg) *err_msg = strdup("Failed to load PTX module");
        return ACE_ERROR_COMPILE;
    }
    
    err = cuModuleGetFunction(&k->func, k->module, name);
    if (err != CUDA_SUCCESS) {
        cuModuleUnload(k->module);
        free(k->name);
        free(k);
        if (err_msg) *err_msg = strdup("Failed to get kernel function");
        return ACE_ERROR_COMPILE;
    }
    
    *kernel = k;
    return ACE_OK;
}

#else

static ace_error_t cuda_kernel_compile(void* dev, const char* name, const char* src,
                                        void** kernel, char** err_msg) {
    if (err_msg) *err_msg = strdup("NVRTC not available");
    return ACE_ERROR_COMPILE;
}

#endif /* NVRTC_AVAILABLE */

static void cuda_kernel_release(void* kernel) {
    cuda_kernel_t* k = (cuda_kernel_t*)kernel;
    if (k) {
        if (k->module) cuModuleUnload(k->module);
        free(k->name);
        free(k);
    }
}

static ace_error_t cuda_kernel_launch(void* kernel, ace_launch_config_t* cfg,
                                       void** args, size_t* sizes, int n) {
    cuda_kernel_t* k = (cuda_kernel_t*)kernel;
    if (!k || !k->func) return ACE_ERROR_LAUNCH;
    
    cuCtxSetCurrent(k->dev->context);
    
    void* kernel_args[16];
    for (int i = 0; i < n && i < 16; i++) {
        kernel_args[i] = args[i];
    }
    
    CUresult err = cuLaunchKernel(
        k->func,
        (unsigned int)cfg->grid[0], (unsigned int)cfg->grid[1], (unsigned int)cfg->grid[2],
        (unsigned int)cfg->block[0], (unsigned int)cfg->block[1], (unsigned int)cfg->block[2],
        (unsigned int)cfg->shared_mem, NULL,
        kernel_args, NULL
    );
    
    return (err == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
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
    .kernel_compile = cuda_kernel_compile,
    .kernel_release = cuda_kernel_release,
    .kernel_launch = cuda_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CUDA, "CUDA", &cuda_ops)

#else

/* CUDA SDK not available - provide stub */
static ace_backend_ops_t cuda_ops = {0};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CUDA, "CUDA (unavailable)", &cuda_ops)

#endif /* CUDA_AVAILABLE */
