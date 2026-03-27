/**
 * @file cuda_device.c
 * @brief CUDA backend device management
 */
#include "cuda_backend.h"

#ifdef CUDA_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvrtc.h>

ace_error_t cuda_init(ace_backend_info_t* info) {
    (void)info;
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

void cuda_shutdown(ace_backend_info_t* info) {
    /* CUDA driver cleanup is automatic */
    (void)info;
}

ace_error_t cuda_device_count(int* count) {
    int c = 0;
    CUresult err = cuDeviceGetCount(&c);
    *count = (err == CUDA_SUCCESS) ? c : 0;
    return ACE_OK;
}

ace_error_t cuda_device_get(int idx, void** dev) {
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

void cuda_device_release(void* dev) {
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

ace_error_t cuda_device_props(void* dev, void* props) {
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

#endif /* CUDA_AVAILABLE */
