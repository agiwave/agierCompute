/**
 * @file opencl_memory.c
 * @brief OpenCL backend memory management
 */
#include "opencl_backend.h"

#ifdef OPENCL_AVAILABLE

#include <stdlib.h>

ace_error_t ocl_mem_alloc(void* dev, size_t size, void** ptr) {
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

void ocl_mem_free(void* dev, void* ptr) {
    (void)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)ptr;
    if (buf) {
        clReleaseMemObject(buf->mem);
        free(buf);
    }
}

ace_error_t ocl_mem_write(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)dst;
    cl_int err = clEnqueueWriteBuffer(d->queue, buf->mem, CL_TRUE, 0, size, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

ace_error_t ocl_mem_read(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)src;
    cl_int err = clEnqueueReadBuffer(d->queue, buf->mem, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

ace_error_t ocl_finish(void* dev) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_DEVICE;
    return (clFinish(d->queue) == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#endif /* OPENCL_AVAILABLE */
