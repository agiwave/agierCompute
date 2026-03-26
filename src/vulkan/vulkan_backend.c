/**
 * @file vulkan_backend.c
 * @brief Vulkan backend - 每设备独立编译和缓存内核
 */
#include "ace.h"
#include "../ace_backend_api.h"

#ifdef VULKAN_AVAILABLE

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef SHADERC_AVAILABLE
#include <shaderc/shaderc.h>
static shaderc_compiler_t g_shaderc = NULL;
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
#define MAX_CMD_BUFFERS 16  /* 命令缓冲池大小 - 增加以支持批处理 */

typedef struct {
    vk_device_t* dev;
    vk_cached_kernel_t kernels[MAX_CACHED_KERNELS];
    int kernel_count;
    VkCommandBuffer cmd_buffers[MAX_CMD_BUFFERS];  /* 命令缓冲池 */
    int cmd_buffer_index;  /* 当前使用的命令缓冲 */
    VkFence fences[MAX_CMD_BUFFERS];  /* 同步 fence */
    VkSemaphore semaphores[MAX_CMD_BUFFERS];  /* 信号量 */
} vk_device_internal_t;

static VkInstance g_instance = VK_NULL_HANDLE;
static int g_initialized = 0;

static vk_cached_kernel_t* find_cached_kernel(vk_device_internal_t* dev_int, const char* name) {
    for (int i = 0; i < dev_int->kernel_count; i++) {
        if (strcmp(dev_int->kernels[i].name, name) == 0) {
            return &dev_int->kernels[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * GLSL 类型名称辅助函数
 * ============================================================================ */

static const char* get_glsl_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "double";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "int64_t";
        case ACE_DTYPE_INT8:     return "int8_t";
        case ACE_DTYPE_UINT8:    return "uint8_t";
        case ACE_DTYPE_INT16:    return "int16_t";
        case ACE_DTYPE_FLOAT16:  return "float16_t";
        default:                 return "float";
    }
}

static int is_float_dtype(ace_dtype_t dtype) {
    return (dtype == ACE_DTYPE_FLOAT32 || dtype == ACE_DTYPE_FLOAT64 || 
            dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16);
}

/* ============================================================================
 * GLSL translation
 * ============================================================================ */

static char* translate_to_glsl(const char* name, const char* src, ace_dtype_t dtype, int* n_buffers, int* n_scalars) {
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");

    size_t body_len = body_end - body_start - 1;
    const char* glsl_type = get_glsl_type_name(dtype);

    typedef struct {
        char name[64];
        int is_buffer;
    } param_info_t;

    param_info_t params[16];
    int n_params = 0;
    *n_buffers = 0;
    *n_scalars = 0;

    const char* p = strchr(src, '(');
    if (p) {
        p++;
        while (*p && *p != ')') {
            while (*p == ' ' || *p == '\t' || *p == '\n') p++;
            if (*p == ')') break;

            const char* star = strchr(p, '*');
            const char* comma = strchr(p, ',');
            const char* paren = strchr(p, ')');
            const char* end = comma ? (paren && paren < comma ? paren : comma) : paren;

            if (star && (!end || star < end)) {
                params[n_params].is_buffer = 1;
                const char* name_start = star + 1;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t')) name_end--;
                size_t len = name_end - name_start;
                if (len >= sizeof(params[n_params].name)) len = sizeof(params[n_params].name) - 1;
                strncpy(params[n_params].name, name_start, len);
                params[n_params].name[len] = '\0';
                (*n_buffers)++;
            } else {
                params[n_params].is_buffer = 0;
                const char* type_end = end;
                while (type_end > p && *(type_end-1) != ' ' && *(type_end-1) != '\t') type_end--;
                if (type_end == p) type_end = end;
                const char* name_start = type_end;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t')) name_end--;
                size_t len = name_end - name_start;
                if (len >= sizeof(params[n_params].name)) len = sizeof(params[n_params].name) - 1;
                strncpy(params[n_params].name, name_start, len);
                params[n_params].name[len] = '\0';
                (*n_scalars)++;
            }
            n_params++;

            if (comma && comma < paren) p = comma + 1;
            else p = end;
        }
    }

    if (*n_buffers == 0) *n_buffers = 1;
    if (*n_buffers > 8) *n_buffers = 8;

    char push_constants[1024] = "";
    char pc_access[1024] = "";

    if (*n_scalars > 0) {
        strcpy(push_constants, "layout(push_constant) uniform PC {\n");
        int scalar_idx = 0;
        for (int i = 0; i < n_params; i++) {
            if (!params[i].is_buffer) {
                /* 第一个标量参数 (n) 总是 int，后续根据数据类型决定 */
                char line[128];
                const char* scalar_type = (scalar_idx == 0) ? "int" : (is_float_dtype(dtype) ? "float" : "int");
                snprintf(line, sizeof(line), "  %s s%d;\n", scalar_type, scalar_idx);
                strcat(push_constants, line);
                char access[128];
                snprintf(access, sizeof(access), "#define %s pc.s%d\n", params[i].name, scalar_idx);
                strcat(pc_access, access);
                scalar_idx++;
            }
        }
        strcat(push_constants, "} pc;\n");
    }

    char buffers[2048] = "";
    int buf_idx = 0;
    for (int i = 0; i < n_params && buf_idx < *n_buffers; i++) {
        if (params[i].is_buffer) {
            char buf_decl[256];
            snprintf(buf_decl, sizeof(buf_decl),
                "layout(binding = %d, std430) buffer B%d { %s d%d[]; };\n",
                buf_idx, buf_idx, glsl_type, buf_idx);
            strcat(buffers, buf_decl);
            char def[128];
            snprintf(def, sizeof(def), "#define %s d%d\n", params[i].name, buf_idx);
            strcat(buffers, def);
            buf_idx++;
        }
    }

    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';

    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access);
    char* out = (char*)malloc(len);

    /* 添加 GLSL 扩展和类型定义 */
    const char* extensions = "";
    const char* type_defs = "";
    if (dtype == ACE_DTYPE_FLOAT64) {
        extensions = "#extension GL_ARB_gpu_shader_fp64 : require\n";
    } else if (dtype == ACE_DTYPE_INT64) {
        extensions = "#extension GL_KHR_shader_subgroup_basic : require\n";
    } else if (dtype == ACE_DTYPE_FLOAT16) {
        extensions = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    }
    
    snprintf(out, len,
        "#version 450\n"
        "%s"
        "%s"
        "layout(local_size_x = 256) in;\n"
        "%s\n"
        "%s\n"
        "%s\n"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "void main() { %s }\n",
        extensions, type_defs,
        buffers, push_constants, pc_access, body);

    free(body);
    return out;
}

static uint32_t find_memory_type(vk_device_t* dev, uint32_t filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < dev->mem_props.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (dev->mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
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

    if (vkCreateInstance(&create_info, NULL, &g_instance) != VK_SUCCESS) {
        printf("[Vulkan] Failed to create instance\n");
        return ACE_ERROR_BACKEND;
    }

#ifdef SHADERC_AVAILABLE
    g_shaderc = shaderc_compiler_initialize();
    printf("[Vulkan] Backend initialized (with shaderc)\n");
#else
    printf("[Vulkan] Backend initialized\n");
#endif
    g_initialized = 1;
    return ACE_OK;
}

static void vk_shutdown(ace_backend_info_t* info) {
#ifdef SHADERC_AVAILABLE
    if (g_shaderc) { shaderc_compiler_release(g_shaderc); g_shaderc = NULL; }
#endif
    if (g_instance) { vkDestroyInstance(g_instance, NULL); g_instance = VK_NULL_HANDLE; }
    g_initialized = 0;
}

static ace_error_t vk_device_count(int* count) {
    if (!g_initialized) { *count = 0; return ACE_OK; }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, NULL);
    *count = (int)n;
    return ACE_OK;
}

static ace_error_t vk_device_get(int idx, void** dev) {
    if (!g_initialized) return ACE_ERROR_DEVICE;

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_instance, &count, NULL);
    if (idx >= (int)count) return ACE_ERROR_DEVICE;

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &count, devices);

    vk_device_internal_t* d = (vk_device_internal_t*)calloc(1, sizeof(*d));
    d->dev = (vk_device_t*)calloc(1, sizeof(vk_device_t));
    d->dev->physical_device = devices[idx];
    vkGetPhysicalDeviceProperties(d->dev->physical_device, &d->dev->props);
    vkGetPhysicalDeviceMemoryProperties(d->dev->physical_device, &d->dev->mem_props);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->dev->physical_device, &qcount, NULL);
    VkQueueFamilyProperties* qprops = (VkQueueFamilyProperties*)malloc(qcount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(d->dev->physical_device, &qcount, qprops);
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d->dev->queue_family = i; break; }
    }
    free(qprops);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
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
        free(d->dev); free(d); free(devices); return ACE_ERROR_DEVICE;
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
        .flags = VK_FENCE_CREATE_SIGNALED_BIT  /* 初始为 signaled 状态 */
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

static void vk_device_release(void* dev) {
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

static ace_error_t vk_device_props(void* dev, void* props) {
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

static ace_error_t vk_mem_alloc(void* dev, size_t size, void** ptr) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    vk_buffer_t* buf = (vk_buffer_t*)calloc(1, sizeof(*buf));
    buf->dev = d->dev;
    buf->size = size;

    vkCreateBuffer(d->dev->device, &buf_info, NULL, &buf->buffer);

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(d->dev->device, buf->buffer, &reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = find_memory_type(d->dev, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    vkAllocateMemory(d->dev->device, &alloc_info, NULL, &buf->memory);
    vkBindBufferMemory(d->dev->device, buf->buffer, buf->memory, 0);

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
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    if (!d || !d->dev || !d->dev->queue) return ACE_ERROR_DEVICE;
    return (vkQueueWaitIdle(d->dev->queue) == VK_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

/* ============================================================================
 * Kernel compile and launch - 每设备独立编译和缓存
 * ============================================================================ */

#ifdef SHADERC_AVAILABLE
static ace_error_t vk_kernel_compile(void* dev, ace_kernel_def_t* kernel_def,
                                      void** kernel, char** err_msg) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;

    /* 计算内核 ID */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;

    /* 查找缓存 */
    for (int i = 0; i < d->kernel_count; i++) {
        if (d->kernels[i].id == kernel_id) {
            *kernel = &d->kernels[i];
            return ACE_OK;
        }
    }

    /* 编译新内核 - 使用 kernel_def 中的 dtype */
    if (d->kernel_count >= MAX_CACHED_KERNELS) {
        if (err_msg) *err_msg = strdup("Kernel cache full");
        return ACE_ERROR_MEM;
    }

    /* 使用 kernel_def 中的数据类型 */
    ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;

    int n_buffers = 0, n_scalars = 0;
    char* glsl = translate_to_glsl(kernel_def->name, kernel_def->src, dtype, &n_buffers, &n_scalars);

    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        g_shaderc, glsl, strlen(glsl), shaderc_compute_shader, kernel_def->name, "main", NULL);

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        if (err_msg) {
            const char* err = shaderc_result_get_error_message(result);
            *err_msg = strdup(err);
        }
        shaderc_result_release(result);
        free(glsl);
        return ACE_ERROR_COMPILE;
    }

    const uint32_t* spirv = (const uint32_t*)shaderc_result_get_bytes(result);
    size_t spirv_size = shaderc_result_get_length(result);

    VkShaderModuleCreateInfo sm_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv
    };
    VkShaderModule shader;
    if (vkCreateShaderModule(d->dev->device, &sm_info, NULL, &shader) != VK_SUCCESS) {
        shaderc_result_release(result);
        free(glsl);
        if (err_msg) *err_msg = strdup("Failed to create shader module");
        return ACE_ERROR_COMPILE;
    }
    shaderc_result_release(result);
    free(glsl);

    /* Create descriptor set layout */
    VkDescriptorSetLayoutBinding* bindings = (VkDescriptorSetLayoutBinding*)calloc(n_buffers, sizeof(VkDescriptorSetLayoutBinding));
    for (int i = 0; i < n_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = n_buffers,
        .pBindings = bindings
    };
    VkDescriptorSetLayout desc_layout;
    if (vkCreateDescriptorSetLayout(d->dev->device, &dsl_info, NULL, &desc_layout) != VK_SUCCESS) {
        free(bindings);
        vkDestroyShaderModule(d->dev->device, shader, NULL);
        if (err_msg) *err_msg = strdup("Failed to create descriptor set layout");
        return ACE_ERROR_COMPILE;
    }
    free(bindings);

    /* Create pipeline layout */
    VkPushConstantRange pc_range = {0};
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &desc_layout
    };
    if (n_scalars > 0) {
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = n_scalars * sizeof(float);
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
    }

    VkPipelineLayout layout;
    vkCreatePipelineLayout(d->dev->device, &pl_info, NULL, &layout);

    /* Create pipeline */
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main"
    };
    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = layout
    };
    VkPipeline pipeline;
    vkCreateComputePipelines(d->dev->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline);

    /* Create descriptor pool and set */
    VkDescriptorPoolSize pool_sizes[8];
    for (int i = 0; i < n_buffers && i < 8; i++) {
        pool_sizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[i].descriptorCount = 1;
    }

    VkDescriptorPoolCreateInfo dp_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = n_buffers,
        .pPoolSizes = pool_sizes
    };
    VkDescriptorPool desc_pool;
    vkCreateDescriptorPool(d->dev->device, &dp_info, NULL, &desc_pool);

    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (n_buffers > 0) {
        VkDescriptorSetAllocateInfo ds_alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &desc_layout
        };
        vkAllocateDescriptorSets(d->dev->device, &ds_alloc, &desc_set);
    }

    /* 缓存内核 */
    vk_cached_kernel_t* k = &d->kernels[d->kernel_count++];
    k->id = kernel_id;  /* 使用计算出的 kernel_id */
    k->name = strdup(kernel_def->name);
    k->src = strdup(kernel_def->src);
    k->shader = shader;
    k->pipeline = pipeline;
    k->layout = layout;
    k->desc_layout = desc_layout;
    k->desc_pool = desc_pool;
    k->desc_set = desc_set;
    k->n_buffers = n_buffers;
    k->n_scalars = n_scalars;

    *kernel = k;
    return ACE_OK;
}
#else
static ace_error_t vk_kernel_compile(void* dev, const char* name, const char* src,
                                      void** kernel, char** err_msg) {
    if (err_msg) *err_msg = strdup("Vulkan requires shaderc");
    return ACE_ERROR_COMPILE;
}
#endif

static void vk_kernel_release(void* kernel) {
    /* Kernels are freed in vk_device_release */
    (void)kernel;
}

static ace_error_t vk_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                     ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    vk_device_internal_t* d_int = (vk_device_internal_t*)dev;
    if (!d_int) return ACE_ERROR_LAUNCH;

    /* 查找缓存的内核 */
    /* 内核 ID 规则：core_id * 16 + dtype，确保不同数据类型有不同缓存 */
    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    vk_cached_kernel_t* k = NULL;
    for (int i = 0; i < d_int->kernel_count; i++) {
        if (d_int->kernels[i].id == kernel_id) {
            k = &d_int->kernels[i];
            break;
        }
    }

    /* 如果未缓存，编译内核 */
    if (!k) {
        /* 调用编译函数 */
        void* compiled_kernel = NULL;
        char* err_msg = NULL;
        ace_error_t err = vk_kernel_compile(dev, kernel_def, &compiled_kernel, &err_msg);
        if (err != ACE_OK) {
            if (err_msg) {
                printf("[Vulkan] Compile error: %s\n", err_msg);
                free(err_msg);
            }
            return ACE_ERROR_COMPILE;
        }
        k = (vk_cached_kernel_t*)compiled_kernel;
    }

    /* 获取数据类型 */
    ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;
    int is_float_type = is_float_dtype(dtype);

    /* Find device from first buffer */
    vk_device_t* vk_dev = NULL;
    for (int i = 0; i < n; i++) {
        if (sizes[i] == ACE_ARG_BUFFER) {
            vk_buffer_t* buf = (vk_buffer_t*)args[i];
            if (buf && buf->dev) { vk_dev = buf->dev; break; }
        }
    }
    if (!vk_dev) return ACE_ERROR_LAUNCH;

    /* 使用静态数组避免 malloc 开销 */
    VkWriteDescriptorSet writes[8];
    VkDescriptorBufferInfo buf_infos[8];
    union { int i; float f; } push_values[8];  /* push constants 值 - 支持 int 和 float */

    /* 更新 descriptor sets */
    int buf_idx = 0;
    if (k->n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        memset(writes, 0, sizeof(writes));

        for (int i = 0; i < n && buf_idx < k->n_buffers && buf_idx < 8; i++) {
            if (sizes[i] == ACE_ARG_BUFFER) {
                vk_buffer_t* buf = (vk_buffer_t*)args[i];
                buf_infos[buf_idx].buffer = buf->buffer;
                buf_infos[buf_idx].offset = 0;
                buf_infos[buf_idx].range = VK_WHOLE_SIZE;

                writes[buf_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[buf_idx].dstSet = k->desc_set;
                writes[buf_idx].dstBinding = buf_idx;
                writes[buf_idx].descriptorCount = 1;
                writes[buf_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[buf_idx].pBufferInfo = &buf_infos[buf_idx];
                buf_idx++;
            }
        }
        vkUpdateDescriptorSets(vk_dev->device, buf_idx, writes, 0, NULL);
    }

    /* Push constants - 第一个参数 n 总是 int，后续参数根据内核类型 */
    int push_count = 0;
    for (int i = 0; i < n && push_count < k->n_scalars && push_count < 8; i++) {
        if (sizes[i] == ACE_ARG_VALUE) {
            /* 第一个标量参数 (n) 总是 int，后续根据内核类型 */
            if (push_count == 0) {
                push_values[push_count].i = *(int*)args[i];
            } else if (is_float_type) {
                push_values[push_count].f = *(float*)args[i];
            } else {
                push_values[push_count].i = *(int*)args[i];
            }
            push_count++;
        }
    }

    /* 使用环形命令缓冲池 */
    int cmd_idx = d_int->cmd_buffer_index;
    VkCommandBuffer cmd = d_int->cmd_buffers[cmd_idx];
    VkFence fence = d_int->fences[cmd_idx];
    
    /* 等待之前的命令完成（带 1 秒超时，避免死锁） */
    VkResult result = vkWaitForFences(vk_dev->device, 1, &fence, VK_TRUE, 1000000000);
    if (result != VK_SUCCESS && result != VK_TIMEOUT) {
        return ACE_ERROR_LAUNCH;
    }
    vkResetFences(vk_dev->device, 1, &fence);
    
    /* 重置命令缓冲 */
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);

    if (k->n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout,
            0, 1, &k->desc_set, 0, NULL);
    }

    if (push_count > 0) {
        /* 计算 push constants 大小：第一个参数是 int，后续根据类型 */
        size_t push_size = 0;
        for (int i = 0; i < push_count; i++) {
            push_size += (i == 0) ? sizeof(int) : (is_float_type ? sizeof(float) : sizeof(int));
        }
        vkCmdPushConstants(cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, push_size, &push_values[0]);
    }

    uint32_t groups = (uint32_t)((cfg->grid[0] * cfg->block[0] + 255) / 256);
    vkCmdDispatch(cmd, groups, 1, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };
    vkQueueSubmit(vk_dev->queue, 1, &submit, fence);
    
    /* 循环使用命令缓冲 */
    d_int->cmd_buffer_index = (cmd_idx + 1) % MAX_CMD_BUFFERS;

    return ACE_OK;
}

/* ============================================================================
 * Backend registration
 * ============================================================================ */

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

static ace_backend_ops_t vk_ops = {0};
ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_VULKAN, "Vulkan (unavailable)", &vk_ops)

#endif
