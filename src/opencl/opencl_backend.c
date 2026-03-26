/**
 * @file opencl_backend.c
 * @brief OpenCL backend entry point
 */
#include "opencl_backend.h"

#ifdef OPENCL_AVAILABLE

/* Backend ops table */
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
