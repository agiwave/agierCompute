/**
 * @file engine_device.c
 * @brief Device management API
 */
#include "engine_internal.h"

#include <stdlib.h>
#include <string.h>

ace_error_t ace_device_count(ace_device_type_t type, int* count) {
    if (!count) return ACE_ERROR_INVALID;
    *count = 0;

    engine_auto_init();

    if (type == ACE_DEVICE_ALL) {
        /* 统计所有类型设备总数 */
        for (int t = 0; t < ACE_DEVICE_COUNT - 1; t++) {
            int c = 0;
            ace_device_count((ace_device_type_t)t, &c);
            *count += c;
        }
        return ACE_OK;
    }

    backend_entry_t* b = engine_find_backend(type);
    if (!b || !b->ops.device_count) {
        *count = 0;
        return ACE_OK;
    }

    return b->ops.device_count(count);
}

ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev) {
    engine_auto_init();

    if (type == ACE_DEVICE_ALL) {
        /* 遍历所有类型查找第 idx 个设备 */
        int global_idx = 0;
        for (int t = 0; t < ACE_DEVICE_COUNT - 1; t++) {
            int count = 0;
            ace_device_count((ace_device_type_t)t, &count);

            if (idx < global_idx + count) {
                return ace_device_get((ace_device_type_t)t, idx - global_idx, dev);
            }
            global_idx += count;
        }
        return ACE_ERROR_NOT_FOUND;
    }

    backend_entry_t* b = engine_find_backend(type);
    if (!b) return ACE_ERROR_NOT_FOUND;

    ace_device_t d = (ace_device_t)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;

    d->backend = b;

    if (b->ops.device_get) {
        ace_error_t err = b->ops.device_get(idx, &d->handle);
        if (err != ACE_OK) {
            free(d);
            return err;
        }
    }

    *dev = d;
    return ACE_OK;
}

void ace_device_release(ace_device_t dev) {
    if (!dev) return;
    if (dev->backend && dev->backend->ops.device_release) {
        dev->backend->ops.device_release(dev->handle);
    }
    free(dev);
}

ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props) {
    if (!dev || !props) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.device_props) return ACE_ERROR_BACKEND;
    return dev->backend->ops.device_props(dev->handle, props);
}
