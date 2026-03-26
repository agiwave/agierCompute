/**
 * @file cuda_backend.c
 * @brief CUDA backend entry point
 */
#include "cuda_backend.h"

#ifdef CUDA_AVAILABLE

/* Backend ops table */
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
