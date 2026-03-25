/**
 * @file vulkan_backend.c
 * @brief Vulkan backend using official Vulkan SDK
 */
#include "ace.h"
#include "../ace_backend_api.h"

#ifdef VULKAN_AVAILABLE

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} vk_device_t;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    vk_device_t* dev;
} vk_buffer_t;

typedef struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool desc_pool;
    char* name;
    vk_device_t* dev;
} vk_kernel_t;

static VkInstance g_instance = VK_NULL_HANDLE;
static int g_initialized = 0;

/* ============================================================================
 * ACE -> GLSL translation
 * ============================================================================ */

static char* translate_to_glsl(const char* name, const char* src) {
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    
    if (!body_start || !body_end) {
        char* out = (char*)malloc(256);
        snprintf(out, 256, "#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
        return out;
    }
    
    size_t body_len = body_end - body_start - 1;
    
    size_t total_len = 512 + body_len;
    char* out = (char*)malloc(total_len);
    
    snprintf(out, total_len,
        "#version 450\n"
        "layout(local_size_x = 256) in;\n"
        "\n"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE int(gl_WorkGroupSize.x)\n"
        "#define BARRIER() barrier()\n"
        "\n"
        "void main() {\n"
        "    %.*s\n"
        "}\n",
        (int)body_len, body_start + 1
    );
    
    return out;
}

/* ============================================================================
 * Helper functions
 * ============================================================================ */

static uint32_t find_memory_type(vk_device_t* dev, uint32_t type_filter, 
                                  VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(dev->physical_device, &mem_props);
    
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

/* ============================================================================
 * Backend operations
 * ============================================================================ */

static ace_error_t vk_init(ace_backend_info_t* info) {
    if (g_initialized) return ACE_OK;
    
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
    
    VkResult result = vkCreateInstance(&create_info, NULL, &g_instance);
    if (result != VK_SUCCESS) {
        printf("[Vulkan] Failed to create instance: %d\n", result);
        return ACE_ERROR_BACKEND;
    }
    
    g_initialized = 1;
    printf("[Vulkan] Backend initialized\n");
    return ACE_OK;
}

static void vk_shutdown(ace_backend_info_t* info) {
    if (g_instance) {
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
    }
    g_initialized = 0;
}

static ace_error_t vk_device_count(int* count) {
    if (!g_initialized) { *count = 0; return ACE_OK; }
    
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(g_instance, &device_count, NULL);
    *count = (int)device_count;
    return ACE_OK;
}

static ace_error_t vk_device_get(int idx, void** dev) {
    if (!g_initialized) return ACE_ERROR_DEVICE;
    
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(g_instance, &device_count, NULL);
    if (idx >= (int)device_count) return ACE_ERROR_DEVICE;
    
    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &device_count, devices);
    
    vk_device_t* d = (vk_device_t*)calloc(1, sizeof(*d));
    d->physical_device = devices[idx];
    
    vkGetPhysicalDeviceProperties(d->physical_device, &d->props);
    
    /* Find queue family */
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties* queue_families = (VkQueueFamilyProperties*)malloc(
        queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &queue_family_count, queue_families);
    
    d->queue_family = 0;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d->queue_family = i;
            break;
        }
    }
    free(queue_families);
    
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info
    };
    
    VkResult result = vkCreateDevice(d->physical_device, &device_info, NULL, &d->device);
    free(devices);
    
    if (result != VK_SUCCESS) {
        free(d);
        return ACE_ERROR_DEVICE;
    }
    
    vkGetDeviceQueue(d->device, d->queue_family, 0, &d->queue);
    
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = d->queue_family
    };
    
    vkCreateCommandPool(d->device, &pool_info, NULL, &d->cmd_pool);
    
    printf("[Vulkan] Device created: %s\n", d->props.deviceName);
    *dev = d;
    return ACE_OK;
}

static void vk_device_release(void* dev) {
    vk_device_t* d = (vk_device_t*)dev;
    if (d) {
        if (d->cmd_pool) vkDestroyCommandPool(d->device, d->cmd_pool, NULL);
        if (d->device) vkDestroyDevice(d->device, NULL);
        free(d);
    }
}

static ace_error_t vk_device_props(void* dev, void* props) {
    vk_device_t* d = (vk_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;
    
    p->type = ACE_DEVICE_VULKAN;
    strncpy(p->name, d->props.deviceName, sizeof(p->name) - 1);
    strcpy(p->vendor, "Vulkan");
    p->total_memory = 0;  /* Would need memory properties */
    p->max_threads = d->props.limits.maxComputeWorkGroupSize[0];
    p->compute_units = 0;
    return ACE_OK;
}

static ace_error_t vk_mem_alloc(void* dev, size_t size, void** ptr) {
    vk_device_t* d = (vk_device_t*)dev;
    
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    vk_buffer_t* buf = (vk_buffer_t*)calloc(1, sizeof(*buf));
    buf->dev = d;
    buf->size = size;
    
    vkCreateBuffer(d->device, &buffer_info, NULL, &buf->buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(d->device, buf->buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(d, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    
    vkAllocateMemory(d->device, &alloc_info, NULL, &buf->memory);
    vkBindBufferMemory(d->device, buf->buffer, buf->memory, 0);
    
    *ptr = buf;
    return ACE_OK;
}

static void vk_mem_free(void* dev, void* ptr) {
    vk_buffer_t* buf = (vk_buffer_t*)ptr;
    if (buf) {
        vkDestroyBuffer(buf->dev->device, buf->buffer, NULL);
        vkFreeMemory(buf->dev->device, buf->memory, NULL);
        free(buf);
    }
}

static ace_error_t vk_mem_write(void* dev, void* dst, const void* src, size_t size) {
    vk_buffer_t* buf = (vk_buffer_t*)dst;
    void* mapped;
    vkMapMemory(buf->dev->device, buf->memory, 0, size, 0, &mapped);
    memcpy(mapped, src, size);
    vkUnmapMemory(buf->dev->device, buf->memory);
    return ACE_OK;
}

static ace_error_t vk_mem_read(void* dev, void* dst, const void* src, size_t size) {
    vk_buffer_t* buf = (vk_buffer_t*)src;
    void* mapped;
    vkMapMemory(buf->dev->device, buf->memory, 0, size, 0, &mapped);
    memcpy(dst, mapped, size);
    vkUnmapMemory(buf->dev->device, buf->memory);
    return ACE_OK;
}

static ace_error_t vk_finish(void* dev) {
    vk_device_t* d = (vk_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_DEVICE;
    return (vkQueueWaitIdle(d->queue) == VK_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

static ace_error_t vk_kernel_compile(void* dev, const char* name, const char* src,
                                      void** kernel, char** err_msg) {
    /* Vulkan requires SPIRV compilation which needs glslang or shaderc */
    /* This is a placeholder for full implementation */
    if (err_msg) *err_msg = _strdup("Vulkan kernel compilation requires glslang/shaderc");
    return ACE_ERROR_COMPILE;
}

static void vk_kernel_release(void* kernel) {
    vk_kernel_t* k = (vk_kernel_t*)kernel;
    if (k) {
        if (k->pipeline) vkDestroyPipeline(k->dev->device, k->pipeline, NULL);
        if (k->layout) vkDestroyPipelineLayout(k->dev->device, k->layout, NULL);
        if (k->desc_layout) vkDestroyDescriptorSetLayout(k->dev->device, k->desc_layout, NULL);
        if (k->desc_pool) vkDestroyDescriptorPool(k->dev->device, k->desc_pool, NULL);
        free(k->name);
        free(k);
    }
}

static ace_error_t vk_kernel_launch(void* kernel, ace_launch_config_t* cfg,
                                     void** args, size_t* sizes, int n) {
    /* Placeholder - would need full descriptor set management */
    return ACE_ERROR_LAUNCH;
}

/* Backend registration */
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
    .kernel_compile = vk_kernel_compile,
    .kernel_release = vk_kernel_release,
    .kernel_launch = vk_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_DEVICE_VULKAN, "Vulkan", &vk_ops)

#else

/* Vulkan SDK not available - provide stub */
static ace_backend_ops_t vk_ops = {0};

ACE_DEFINE_BACKEND(ACE_DEVICE_VULKAN, "Vulkan (unavailable)", &vk_ops)

#endif /* VULKAN_AVAILABLE */