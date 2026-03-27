/**
 * @file cuda_kernel.c
 * @brief CUDA backend kernel compilation and execution
 */
#include "cuda_backend.h"

#ifdef NVRTC_AVAILABLE
#include <nvrtc.h>
#endif

#ifdef CUDA_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NVRTC_AVAILABLE

static ace_error_t compile_and_cache_kernel(cuda_device_t* d, ace_kernel_def_t* kernel_def,
                                             int kernel_id, int bucket, cuda_kernel_t** out_kernel) {
    ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;
    char* cuda_src = cuda_translate_code(d, kernel_def->name, kernel_def->src, dtype);
    
    /* 调试输出：打印生成的 CUDA 代码 */
    printf("[CUDA] Generated code for %s:\n---\n%s\n---\n", kernel_def->name, cuda_src);

    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, cuda_src, kernel_def->name, 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        free(cuda_src);
        return ACE_ERROR_COMPILE;
    }

    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "-arch=compute_%d%d", d->compute_major, d->compute_minor);

    const char* opts[16];
    int opt_count = 4;
    opts[0] = arch_opt;
    opts[1] = "-default-device";
    opts[2] = "--std=c++11";
#ifdef _WIN32
    opts[3] = "--include-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include";
#else
    opts[3] = "--include-path=/usr/include";
#endif

    /* 为 FP16/BF16 启用半精度支持 */
    if (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16) {
        opts[opt_count++] = "--fmad=true";
        opts[opt_count++] = "--use_fast_math";

        int arch_version = d->compute_major * 10 + d->compute_minor;
        if (arch_version >= 60) {
            char arch_def[64];
            snprintf(arch_def, sizeof(arch_def), "-D__CUDA_ARCH__=%d%d", d->compute_major, d->compute_minor);
            opts[opt_count++] = arch_def;
        } else if (arch_version >= 50) {
            char arch_def[64];
            snprintf(arch_def, sizeof(arch_def), "-D__CUDA_ARCH__=%d0", d->compute_major);
            opts[opt_count++] = arch_def;
        }
    }

    res = nvrtcCompileProgram(prog, opt_count, opts);
    if (res != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = (char*)malloc(log_size);
        nvrtcGetProgramLog(prog, log);
        printf("[CUDA] Compile error for %s (%s):\n%s\n", kernel_def->name,
               cuda_get_type_name(d, dtype), log);
        free(log);
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
    *out_kernel = k;
    return ACE_OK;
}

#endif /* NVRTC_AVAILABLE */

ace_error_t cuda_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;

    cuCtxSetCurrent(d->context);

    /* 查找缓存的内核 */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    int bucket = kernel_id % KERNEL_CACHE_SIZE;
    cuda_kernel_t* cached = d->kernel_cache.buckets[bucket];
    while (cached) {
        if (cached->id == kernel_id) {
            break;
        }
        cached = cached->next;
    }

    /* 如果未缓存，编译内核 */
    if (!cached) {
#ifdef NVRTC_AVAILABLE
        ace_error_t err = compile_and_cache_kernel(d, kernel_def, kernel_id, bucket, &cached);
        if (err != ACE_OK) {
            return err;
        }
#else
        return ACE_ERROR_COMPILE;
#endif
    }

    /* 执行内核 */
    void* kernel_args[16];
    CUdeviceptr ptr_values[16];

    for (int i = 0; i < n && i < 16; i++) {
        if (sizes[i] <= 0) {
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

#endif /* CUDA_AVAILABLE */
