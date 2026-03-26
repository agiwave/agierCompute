/**
 * @file engine_kernel.c
 * @brief Kernel management API
 */
#include "engine_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global kernel templates */
kernel_template_t g_templates[MAX_TEMPLATES];
int g_template_count = 0;

ace_kernel_t ace_kernel_register(const char* name, const char* src) {
    if (g_template_count >= MAX_TEMPLATES) return NULL;

    /* 检查是否已注册 */
    for (int i = 0; i < g_template_count; i++) {
        if (strcmp(g_templates[i].name, name) == 0) {
            return (ace_kernel_t)(intptr_t)(i + 1);
        }
    }

    g_templates[g_template_count].name = strdup(name);
    g_templates[g_template_count].src = strdup(src);
    g_template_count++;

    return (ace_kernel_t)(intptr_t)g_template_count;
}

kernel_template_t* engine_get_template(ace_kernel_t kernel) {
    int idx = (int)(intptr_t)kernel - 1;
    if (idx < 0 || idx >= g_template_count) return NULL;
    return &g_templates[idx];
}

ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                               ace_dtype_t dtype, size_t n,
                               void** args, int* sizes, int nargs) {
    if (!dev || !kernel) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.kernel_launch) {
        return ACE_ERROR_BACKEND;
    }

    kernel_template_t* tmpl = engine_get_template(kernel);
    if (!tmpl) return ACE_ERROR_COMPILE;

    /* 处理参数，找到第一个 buffer 所属的设备 */
    void* processed_args[16];
    size_t arg_sizes[16];
    ace_device_t actual_dev = dev;

    if (nargs > 16) nargs = 16;
    for (int i = 0; i < nargs; i++) {
        if (sizes[i] <= 0) {
            struct ace_buffer_* buf = (struct ace_buffer_*)args[i];
            processed_args[i] = buf ? buf->ptr : NULL;
            arg_sizes[i] = 0;
            if (buf && buf->dev && actual_dev == dev) {
                actual_dev = buf->dev;
            }
        } else {
            processed_args[i] = args[i];
            arg_sizes[i] = (size_t)sizes[i];
        }
    }

    ace_kernel_def_t kernel_def;
    kernel_def.id = (int)(intptr_t)kernel;
    kernel_def.name = tmpl->name;
    kernel_def.src = tmpl->src;
    kernel_def.dtype = (int)dtype;

    ace_launch_config_t cfg = ace_launch_1d(n, 256);
    ace_error_t err = actual_dev->backend->ops.kernel_launch(actual_dev->handle, &kernel_def, &cfg, processed_args, arg_sizes, nargs);
    if (err != ACE_OK) {
        printf("[ACE] kernel_launch failed: err=%d dtype=%d\n", err, dtype);
    }
    return err;
}

ace_error_t ace_kernel_launch(ace_device_t dev, ace_kernel_t kernel,
                               ace_dtype_t dtype, ace_launch_config_t* config,
                               void** args, int* sizes, int nargs) {
    if (!dev || !kernel) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.kernel_launch) {
        return ACE_ERROR_BACKEND;
    }

    kernel_template_t* tmpl = engine_get_template(kernel);
    if (!tmpl) return ACE_ERROR_COMPILE;

    void* processed_args[16];
    size_t arg_sizes[16];
    ace_device_t actual_dev = dev;

    if (nargs > 16) nargs = 16;
    for (int i = 0; i < nargs; i++) {
        if (sizes[i] <= 0) {
            struct ace_buffer_* buf = (struct ace_buffer_*)args[i];
            processed_args[i] = buf ? buf->ptr : NULL;
            arg_sizes[i] = 0;
            if (buf && buf->dev && actual_dev == dev) {
                actual_dev = buf->dev;
            }
        } else {
            processed_args[i] = args[i];
            arg_sizes[i] = (size_t)sizes[i];
        }
    }

    ace_kernel_def_t kernel_def;
    kernel_def.id = (int)(intptr_t)kernel;
    kernel_def.name = tmpl->name;
    kernel_def.src = tmpl->src;
    kernel_def.dtype = (int)dtype;

    ace_launch_config_t default_cfg = ace_launch_1d(1, 1);
    return actual_dev->backend->ops.kernel_launch(actual_dev->handle, &kernel_def,
                                            config ? config : &default_cfg,
                                            processed_args, arg_sizes, nargs);
}
