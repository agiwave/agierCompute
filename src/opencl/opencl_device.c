/**
 * @file opencl_device.c
 * @brief OpenCL backend device management
 */
#include "opencl_backend.h"

#ifdef OPENCL_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cl_platform_id g_opencl_platform;

/* 设备扩展支持状态 - 全局变量 */
ocl_device_extensions_t g_device_exts = {0};

/* 检查设备扩展支持 */
static void check_device_extensions(cl_device_id device) {
    char extensions[2048] = "";
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    
    g_device_exts.has_fp16 = (strstr(extensions, "cl_khr_fp16") != NULL);
    g_device_exts.has_fp64 = (strstr(extensions, "cl_khr_fp64") != NULL);
    g_device_exts.has_int64 = 1;  /* OpenCL 1.0+ 都支持 int64 */
    
    printf("[OpenCL] Device extensions: FP16=%s, FP64=%s, INT64=%s\n",
           g_device_exts.has_fp16 ? "YES" : "NO",
           g_device_exts.has_fp64 ? "YES" : "NO",
           g_device_exts.has_int64 ? "YES" : "NO");
}

ace_error_t ocl_init(ace_backend_info_t* info) {
    cl_uint num_platforms;
    if (clGetPlatformIDs(1, &g_opencl_platform, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        printf("[OpenCL] No platforms found\n");
        return ACE_ERROR_BACKEND;
    }

    char name[128];
    clGetPlatformInfo(g_opencl_platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
    printf("[OpenCL] Backend initialized (platform: %s)\n", name);
    return ACE_OK;
}

void ocl_shutdown(ace_backend_info_t* info) {
    (void)info;
    /* OpenCL cleanup is automatic */
}

ace_error_t ocl_device_count(int* count) {
    cl_uint num = 0;
    cl_int err = clGetDeviceIDs(g_opencl_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(g_opencl_platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num);
        if (err != CL_SUCCESS) num = 0;
    }
    *count = num;
    return ACE_OK;
}

ace_error_t ocl_device_get(int idx, void** dev) {
    cl_device_id devices[16];
    cl_uint num;

    cl_int err = clGetDeviceIDs(g_opencl_platform, CL_DEVICE_TYPE_GPU, 16, devices, &num);
    if (err != CL_SUCCESS || idx >= (int)num) {
        err = clGetDeviceIDs(g_opencl_platform, CL_DEVICE_TYPE_CPU, 16, devices, &num);
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

    /* 检查设备扩展支持 */
    check_device_extensions(d->device);

    /* 初始化内核缓存 */
    memset(&d->kernel_cache, 0, sizeof(d->kernel_cache));

    *dev = d;
    return ACE_OK;
}

void ocl_device_release(void* dev) {
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

ace_error_t ocl_device_props(void* dev, void* props) {
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

#endif /* OPENCL_AVAILABLE */
