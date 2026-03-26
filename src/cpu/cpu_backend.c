/**
 * @file cpu_backend.c
 * @brief CPU 后端 - 占位实现
 *
 * 注意：完整的 CPU 后端需要 GCC JIT 编译能力
 * 当前版本仅作为框架示例，实际使用时请启用其他后端
 */
#include "ace.h"
#include "../ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * 内部结构
 * ============================================================================ */

typedef struct {
    ace_device_props_t props;
} cpu_device_t;

/* ============================================================================
 * 后端操作实现
 * ============================================================================ */

static ace_error_t cpu_init(ace_backend_info_t* info) {
    (void)info;
    printf("[CPU] Backend loaded (placeholder - use GPU backends for actual execution)\n");
    return ACE_OK;
}

static void cpu_shutdown(ace_backend_info_t* info) {
    (void)info;
}

static ace_error_t cpu_device_count(int* count) {
    /* CPU 后端未实现，返回 0 个设备 */
    *count = 0;
    return ACE_OK;
}

static ace_error_t cpu_device_get(int idx, void** dev) {
    if (idx != 0) return ACE_ERROR_DEVICE;

    cpu_device_t* d = (cpu_device_t*)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;

    strcpy(d->props.name, "CPU (placeholder)");
    strcpy(d->props.vendor, "Generic");
    d->props.type = ACE_DEVICE_CPU;
    d->props.max_threads = 4;
    d->props.compute_units = 4;
    d->props.total_memory = 0;

    *dev = d;
    return ACE_OK;
}

static void cpu_device_release(void* dev) {
    free(dev);
}

static ace_error_t cpu_device_props(void* dev, void* props) {
    if (!dev || !props) return ACE_ERROR_DEVICE;
    cpu_device_t* d = (cpu_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    *p = d->props;
    return ACE_OK;
}

static ace_error_t cpu_mem_alloc(void* dev, size_t size, void** ptr) {
    (void)dev;
    *ptr = calloc(1, size);
    return *ptr ? ACE_OK : ACE_ERROR_MEM;
}

static void cpu_mem_free(void* dev, void* ptr) {
    (void)dev;
    free(ptr);
}

static ace_error_t cpu_mem_write(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_mem_read(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_finish(void* dev) {
    (void)dev;
    return ACE_OK;
}

static ace_error_t cpu_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                      ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    (void)dev; (void)kernel_def; (void)cfg; (void)args; (void)sizes; (void)n;
    printf("[CPU] Warning: Kernel execution requires JIT compilation (not implemented)\n");
    return ACE_ERROR_LAUNCH;
}

/* ============================================================================
 * 后端注册
 * ============================================================================ */

static ace_backend_ops_t cpu_ops = {
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .device_count = cpu_device_count,
    .device_get = cpu_device_get,
    .device_release = cpu_device_release,
    .device_props = cpu_device_props,
    .mem_alloc = cpu_mem_alloc,
    .mem_free = cpu_mem_free,
    .mem_write = cpu_mem_write,
    .mem_read = cpu_mem_read,
    .finish = cpu_finish,
    .kernel_launch = cpu_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CPU, "CPU (placeholder)", &cpu_ops)
