/**
 * @file opencl_kernel.c
 * @brief OpenCL backend kernel compilation and execution
 */
#include "opencl_backend.h"

#ifdef OPENCL_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ace_error_t ocl_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                               ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_LAUNCH;

    /* 查找缓存的内核 */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    int bucket = kernel_id % KERNEL_CACHE_SIZE;
    ocl_kernel_t* cached = d->kernel_cache.buckets[bucket];
    while (cached) {
        if (cached->id == kernel_id) {
            break;
        }
        cached = cached->next;
    }

    /* 如果未缓存，编译内核 */
    if (!cached) {
        ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;
        char* translated = ocl_translate_code(kernel_def->name, kernel_def->src, dtype);
        
        /* 调试输出：打印生成的 OpenCL 代码 */
        printf("[OpenCL] Generated code for %s:\n---\n%s\n---\n", kernel_def->name, translated);

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
            printf("[OpenCL] Compile error for %s (%s):\n%s\n", kernel_def->name,
                   ocl_get_type_name(dtype), log);
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
        if (sizes[i] <= 0) {
            ocl_buffer_t* buf = (ocl_buffer_t*)args[i];
            if (!buf || !buf->mem) {
                return ACE_ERROR_LAUNCH;
            }
            clSetKernelArg(cached->kernel, i, sizeof(cl_mem), &buf->mem);
        } else {
            /* 根据参数大小传递正确的值 */
            clSetKernelArg(cached->kernel, i, sizes[i], args[i]);
        }
    }

    size_t global[3] = {
        cfg->grid[0] * cfg->block[0],
        cfg->grid[1] * cfg->block[1],
        cfg->grid[2] * cfg->block[2]
    };
    size_t local[3] = { cfg->block[0], cfg->block[1], cfg->block[2] };

    cl_int err = clEnqueueNDRangeKernel(d->queue, cached->kernel, 3, NULL, global, local, 0, NULL, NULL);

    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#endif /* OPENCL_AVAILABLE */
