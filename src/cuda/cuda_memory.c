/**
 * @file cuda_memory.c
 * @brief CUDA backend memory management
 */
#include "cuda_backend.h"

#ifdef CUDA_AVAILABLE

#include <stdlib.h>

ace_error_t cuda_mem_alloc(void* dev, size_t size, void** ptr) {
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

void cuda_mem_free(void* dev, void* ptr) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)ptr;
    if (buf) {
        if (buf->ptr) cuMemFree(buf->ptr);
        free(buf);
    }
}

ace_error_t cuda_mem_write(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)dst;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyHtoD(buf->ptr, src, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

ace_error_t cuda_mem_read(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    cuda_buffer_t* buf = (cuda_buffer_t*)src;
    if (!buf || !buf->ptr) return ACE_ERROR_DEVICE;
    return (cuMemcpyDtoH(dst, buf->ptr, size) == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

ace_error_t cuda_finish(void* dev) {
    cuda_device_t* d = (cuda_device_t*)dev;
    if (!d) return ACE_ERROR_DEVICE;
    cuCtxSetCurrent(d->context);
    return (cuCtxSynchronize() == CUDA_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#endif /* CUDA_AVAILABLE */
