/**
 * @file engine_memory.c
 * @brief Memory management API
 */
#include "engine_internal.h"

#include <stdlib.h>

ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf) {
    if (!dev || !buf) return ACE_ERROR_INVALID;

    ace_buffer_t b = (ace_buffer_t)calloc(1, sizeof(*b));
    if (!b) return ACE_ERROR_MEM;

    b->dev = dev;
    b->size = size;

    if (dev->backend && dev->backend->ops.mem_alloc) {
        ace_error_t err = dev->backend->ops.mem_alloc(dev->handle, size, &b->ptr);
        if (err != ACE_OK) {
            free(b);
            return err;
        }
    }

    *buf = b;
    return ACE_OK;
}

void ace_buffer_free(ace_buffer_t buf) {
    if (!buf) return;
    if (buf->dev && buf->dev->backend && buf->dev->backend->ops.mem_free) {
        buf->dev->backend->ops.mem_free(buf->dev->handle, buf->ptr);
    }
    free(buf);
}

ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size) {
    if (!buf || !data || !buf->dev) return ACE_ERROR_INVALID;
    if (!buf->dev->backend || !buf->dev->backend->ops.mem_write) return ACE_ERROR_BACKEND;
    return buf->dev->backend->ops.mem_write(buf->dev->handle, buf->ptr, data, size);
}

ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size) {
    if (!buf || !data || !buf->dev) return ACE_ERROR_INVALID;
    if (!buf->dev->backend || !buf->dev->backend->ops.mem_read) return ACE_ERROR_BACKEND;
    return buf->dev->backend->ops.mem_read(buf->dev->handle, data, buf->ptr, size);
}

ace_error_t ace_finish(ace_device_t dev) {
    if (!dev) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.finish) return ACE_ERROR_BACKEND;
    return dev->backend->ops.finish(dev->handle);
}
