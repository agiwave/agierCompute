/**
 * @file vulkan_backend.h
 * @brief Vulkan backend internal header
 */
#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include "ace.h"
#include "../ace_backend_api.h"
#include "vulkan_dtype_table.h"

#ifdef VULKAN_AVAILABLE

#include <vulkan/vulkan.h>

#ifdef SHADERC_AVAILABLE
#include <shaderc/shaderc.h>
#endif

/* ============================================================================
 * Internal structures
 * ============================================================================ */

typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    uint32_t queue_family;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
} vk_device_t;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    vk_device_t* dev;
} vk_buffer_t;

/* 每设备缓存的内核 */
typedef struct {
    int id;                /* 内核 ID (core_id * 16 + dtype) */
    char* name;
    char* src;
    VkShaderModule shader;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
    int n_buffers;
    int n_scalars;
} vk_cached_kernel_t;

#define MAX_CACHED_KERNELS 64
#define MAX_CMD_BUFFERS 16

typedef struct {
    vk_device_t* dev;
    vk_cached_kernel_t kernels[MAX_CACHED_KERNELS];
    int kernel_count;
    VkCommandBuffer cmd_buffers[MAX_CMD_BUFFERS];
    int cmd_buffer_index;
    VkFence fences[MAX_CMD_BUFFERS];
    VkSemaphore semaphores[MAX_CMD_BUFFERS];
    vk_dtype_table_t dtype_table;  /* 设备级别的类型表 */
} vk_device_internal_t;

/* ============================================================================
 * Type utilities (vulkan_type_utils.c)
 * ============================================================================ */

/* 检测设备特性 */
void vk_detect_device_features(VkPhysicalDevice physical_device, vk_device_features_t* features);

/* 设备级别的类型信息访问 */
static inline const char* vk_get_type_name(const vk_device_internal_t* dev, ace_dtype_t dtype) {
    return vk_dtype_info(&dev->dtype_table, dtype)->name;
}

char* vk_translate_to_glsl(const vk_device_internal_t* dev, const char* name, const char* src, 
                           ace_dtype_t dtype, int* n_buffers, int* n_scalars);

/* ============================================================================
 * Device management (vulkan_device.c)
 * ============================================================================ */

ace_error_t vk_init(ace_backend_info_t* info);
void vk_shutdown(ace_backend_info_t* info);
ace_error_t vk_device_count(int* count);
ace_error_t vk_device_get(int idx, void** dev);
void vk_device_release(void* dev);
ace_error_t vk_device_props(void* dev, void* props);

/* ============================================================================
 * Memory management (vulkan_memory.c)
 * ============================================================================ */

ace_error_t vk_mem_alloc(void* dev, size_t size, void** ptr);
void vk_mem_free(void* dev, void* ptr);
ace_error_t vk_mem_write(void* dev, void* dst, const void* src, size_t size);
ace_error_t vk_mem_read(void* dev, void* dst, const void* src, size_t size);
ace_error_t vk_finish(void* dev);

/* ============================================================================
 * Kernel management (vulkan_kernel.c)
 * ============================================================================ */

ace_error_t vk_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                              ace_launch_config_t* cfg, void** args, size_t* sizes, int n);

/* ============================================================================
 * Global state
 * ============================================================================ */

extern VkInstance g_vk_instance;
extern int g_vk_initialized;
#ifdef SHADERC_AVAILABLE
extern shaderc_compiler_t g_shaderc_compiler;
#endif

/* ============================================================================
 * Internal utilities
 * ============================================================================ */

uint32_t vk_find_memory_type(vk_device_t* dev, uint32_t filter, VkMemoryPropertyFlags props);

#endif /* VULKAN_AVAILABLE */

#endif /* VULKAN_BACKEND_H */
