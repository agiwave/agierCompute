/**
 * @file vulkan_device.c
 * @brief Vulkan backend device management
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

VkInstance g_vk_instance = VK_NULL_HANDLE;
int g_vk_initialized = 0;
#ifdef SHADERC_AVAILABLE
shaderc_compiler_t g_shaderc_compiler = NULL;
#endif

ace_error_t vk_init(ace_backend_info_t* info) {
    if (g_vk_initialized) return ACE_OK;

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "AgierCompute",
        .applicationVersion = 1,
        .pEngineName = "ACE",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_0
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    if (vkCreateInstance(&create_info, NULL, &g_vk_instance) != VK_SUCCESS) {
        printf("[Vulkan] Failed to create instance\n");
        return ACE_ERROR_BACKEND;
    }

#ifdef SHADERC_AVAILABLE
    g_shaderc_compiler = shaderc_compiler_initialize();
    printf("[Vulkan] Backend initialized (with shaderc)\n");
#else
    printf("[Vulkan] Backend initialized\n");
#endif
    g_vk_initialized = 1;
    return ACE_OK;
}

void vk_shutdown(ace_backend_info_t* info) {
    (void)info;
#ifdef SHADERC_AVAILABLE
    if (g_shaderc_compiler) {
        shaderc_compiler_release(g_shaderc_compiler);
        g_shaderc_compiler = NULL;
    }
#endif
    if (g_vk_instance) {
        vkDestroyInstance(g_vk_instance, NULL);
        g_vk_instance = VK_NULL_HANDLE;
    }
    g_vk_initialized = 0;
}

ace_error_t vk_device_count(int* count) {
    if (!g_vk_initialized) {
        *count = 0;
        return ACE_OK;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_vk_instance, &n, NULL);
    *count = (int)n;
    return ACE_OK;
}

ace_error_t vk_device_get(int idx, void** dev) {
    if (!g_vk_initialized) return ACE_ERROR_DEVICE;

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_vk_instance, &count, NULL);
    if (idx >= (int)count) return ACE_ERROR_DEVICE;

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_vk_instance, &count, devices);

    vk_device_internal_t* d = (vk_device_internal_t*)calloc(1, sizeof(*d));
    d->dev = (vk_device_t*)calloc(1, sizeof(vk_device_t));
    d->dev->physical_device = devices[idx];
    vkGetPhysicalDeviceProperties(d->dev->physical_device, &d->dev->props);
    vkGetPhysicalDeviceMemoryProperties(d->dev->physical_device, &d->dev->mem_props);

    /* 检测设备特性 */
    vk_detect_device_features(d->dev->physical_device);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->dev->physical_device, &qcount, NULL);
    VkQueueFamilyProperties* qprops = (VkQueueFamilyProperties*)malloc(qcount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(d->dev->physical_device, &qcount, qprops);
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d->dev->queue_family = i;
            break;
        }
    }
    free(qprops);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueFamilyIndex = d->dev->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qinfo
    };

    if (vkCreateDevice(d->dev->physical_device, &dev_info, NULL, &d->dev->device) != VK_SUCCESS) {
        free(d->dev);
        free(d);
        free(devices);
        return ACE_ERROR_DEVICE;
    }

    vkGetDeviceQueue(d->dev->device, d->dev->queue_family, 0, &d->dev->queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = d->dev->queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    vkCreateCommandPool(d->dev->device, &pool_info, NULL, &d->dev->cmd_pool);

    d->kernel_count = 0;
    d->cmd_buffer_index = 0;

    /* 预分配命令缓冲池 */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = d->dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_CMD_BUFFERS
    };
    vkAllocateCommandBuffers(d->dev->device, &cmd_alloc, d->cmd_buffers);

    /* 创建 fence 和 semaphore */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    VkSemaphoreCreateInfo sem_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    for (int i = 0; i < MAX_CMD_BUFFERS; i++) {
        vkCreateFence(d->dev->device, &fence_info, NULL, &d->fences[i]);
        vkCreateSemaphore(d->dev->device, &sem_info, NULL, &d->semaphores[i]);
    }

    printf("[Vulkan] Device: %s\n", d->dev->props.deviceName);
    free(devices);
    *dev = d;
    return ACE_OK;
}

void vk_device_release(void* dev) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    if (d) {
        /* Free command buffers */
        vkFreeCommandBuffers(d->dev->device, d->dev->cmd_pool, MAX_CMD_BUFFERS, d->cmd_buffers);

        /* Free fences and semaphores */
        for (int i = 0; i < MAX_CMD_BUFFERS; i++) {
            vkDestroyFence(d->dev->device, d->fences[i], NULL);
            vkDestroySemaphore(d->dev->device, d->semaphores[i], NULL);
        }

        /* Free cached kernels */
        for (int i = 0; i < d->kernel_count; i++) {
            vk_cached_kernel_t* k = &d->kernels[i];
            if (k->pipeline) vkDestroyPipeline(d->dev->device, k->pipeline, NULL);
            if (k->layout) vkDestroyPipelineLayout(d->dev->device, k->layout, NULL);
            if (k->desc_layout) vkDestroyDescriptorSetLayout(d->dev->device, k->desc_layout, NULL);
            if (k->shader) vkDestroyShaderModule(d->dev->device, k->shader, NULL);
            if (k->desc_pool) vkDestroyDescriptorPool(d->dev->device, k->desc_pool, NULL);
            free(k->name);
            free(k->src);
        }
        vkDestroyCommandPool(d->dev->device, d->dev->cmd_pool, NULL);
        vkDestroyDevice(d->dev->device, NULL);
        free(d->dev);
        free(d);
    }
}

ace_error_t vk_device_props(void* dev, void* props) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;

    p->type = ACE_BACKEND_DEVICE_VULKAN;
    strncpy(p->name, d->dev->props.deviceName, sizeof(p->name) - 1);
    strcpy(p->vendor, "Vulkan");
    p->total_memory = d->dev->mem_props.memoryHeaps[0].size;
    p->max_threads = d->dev->props.limits.maxComputeWorkGroupSize[0];
    return ACE_OK;
}

#endif /* VULKAN_AVAILABLE */
