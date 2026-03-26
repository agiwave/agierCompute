/**
 * @file vulkan_backend.c
 * @brief Vulkan backend entry point
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

/* Backend ops table */
static ace_backend_ops_t vk_ops = {
    .init = vk_init,
    .shutdown = vk_shutdown,
    .device_count = vk_device_count,
    .device_get = vk_device_get,
    .device_release = vk_device_release,
    .device_props = vk_device_props,
    .mem_alloc = vk_mem_alloc,
    .mem_free = vk_mem_free,
    .mem_write = vk_mem_write,
    .mem_read = vk_mem_read,
    .finish = vk_finish,
    .kernel_launch = vk_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_VULKAN, "Vulkan", &vk_ops)

#else

/* Vulkan SDK not available - provide stub */
static ace_backend_ops_t vk_ops = {0};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_VULKAN, "Vulkan (unavailable)", &vk_ops)

#endif /* VULKAN_AVAILABLE */
