/**
 * @file vulkan_backend.c
 * @brief Vulkan backend using official Vulkan SDK with shaderc for SPIRV compilation
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

/* Internal structures */
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

typedef struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
    int n_buffers;
    int n_scalars;
    char* name;
    vk_device_t* dev;
} vk_kernel_t;

static VkInstance g_instance = VK_NULL_HANDLE;
static int g_initialized = 0;

/* GLSL translation - convert ACE kernel to GLSL compute shader */
static char* translate_to_glsl(const char* name, const char* src, ace_dtype_t dtype, int* n_buffers, int* n_scalars) {
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    
    size_t body_len = body_end - body_start - 1;
    const char* type_name = ace_dtype_name(dtype);
    
    /* Parse parameters */
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
                /* Buffer parameter */
                params[n_params].is_buffer = 1;
                const char* name_start = star + 1;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t' || *(name_end-1) == ',')) name_end--;
                size_t len = name_end - name_start;
                if (len >= sizeof(params[n_params].name)) len = sizeof(params[n_params].name) - 1;
                strncpy(params[n_params].name, name_start, len);
                params[n_params].name[len] = '\0';
                (*n_buffers)++;
            } else {
                /* Scalar parameter */
                params[n_params].is_buffer = 0;
                const char* type_end = end;
                while (type_end > p && *(type_end-1) != ' ' && *(type_end-1) != '\t') type_end--;
                if (type_end == p) type_end = end;
                const char* name_start = type_end;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t' || *(name_end-1) == ',')) name_end--;
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
    
    /* Build push constants block for scalar params */
    char push_constants[1024] = "";
    char pc_access[1024] = "";
    
    if (*n_scalars > 0) {
        strcpy(push_constants, "layout(push_constant) uniform PC {\n");
        int scalar_idx = 0;
        for (int i = 0; i < n_params; i++) {
            if (!params[i].is_buffer) {
                char line[128];
                /* Use float for better precision, cast in shader if needed */
                snprintf(line, sizeof(line), "  float s%d;  /* %s */\n", scalar_idx, params[i].name);
                strcat(push_constants, line);
                
                char access[128];
                snprintf(access, sizeof(access), "#define %s pc.s%d\n", params[i].name, scalar_idx);
                strcat(pc_access, access);
                scalar_idx++;
            }
        }
        strcat(push_constants, "} pc;\n");
    }
    
    /* Build buffer declarations */
    char buffers[2048] = "";
    int buf_idx = 0;
    
    for (int i = 0; i < n_params && buf_idx < *n_buffers; i++) {
        if (params[i].is_buffer) {
            char buf_decl[256];
            snprintf(buf_decl, sizeof(buf_decl), 
                "layout(binding = %d, std430) buffer B%d { %s d%d[]; };\n",
                buf_idx, buf_idx, type_name, buf_idx);
            strcat(buffers, buf_decl);
            
            char def[128];
            snprintf(def, sizeof(def), "#define %s d%d\n", params[i].name, buf_idx);
            strcat(buffers, def);
            buf_idx++;
        }
    }
    
    /* Build processed body */
    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';
    
    /* Assemble final GLSL */
    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) + body_len;
    char* out = (char*)malloc(len);
    snprintf(out, len,
        "#version 450\n"
        "layout(local_size_x = 256) in;\n"
        "%s\n"
        "%s\n"
        "%s\n"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "\n"
        "void main() {\n"
        "  %s\n"
        "}\n",
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

/* Backend operations */
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
    
    vk_device_t* d = (vk_device_t*)calloc(1, sizeof(*d));
    d->physical_device = devices[idx];
    vkGetPhysicalDeviceProperties(d->physical_device, &d->props);
    vkGetPhysicalDeviceMemoryProperties(d->physical_device, &d->mem_props);
    
    /* Find compute queue */
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &qcount, NULL);
    VkQueueFamilyProperties* qprops = (VkQueueFamilyProperties*)malloc(qcount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &qcount, qprops);
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d->queue_family = i; break; }
    }
    free(qprops);
    
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qinfo
    };

    VkResult device_result = vkCreateDevice(d->physical_device, &dev_info, NULL, &d->device);
    if (device_result != VK_SUCCESS || d->device == VK_NULL_HANDLE) {
        fprintf(stderr, "[Vulkan] Failed to create logical device: %d\n", device_result);
        free(d); free(devices); return ACE_ERROR_DEVICE;
    }

    vkGetDeviceQueue(d->device, d->queue_family, 0, &d->queue);
    if (d->queue == VK_NULL_HANDLE) {
        fprintf(stderr, "[Vulkan] Failed to get device queue\n");
        vkDestroyDevice(d->device, NULL);
        free(d); free(devices); return ACE_ERROR_DEVICE;
    }

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = d->queue_family
    };
    VkResult pool_result = vkCreateCommandPool(d->device, &pool_info, NULL, &d->cmd_pool);
    if (pool_result != VK_SUCCESS || d->cmd_pool == VK_NULL_HANDLE) {
        fprintf(stderr, "[Vulkan] Failed to create command pool: %d\n", pool_result);
        vkDestroyDevice(d->device, NULL);
        free(d); free(devices); return ACE_ERROR_DEVICE;
    }

    printf("[Vulkan] Device: %s\n", d->props.deviceName);
    free(devices);
    *dev = d;
    return ACE_OK;
}

static void vk_device_release(void* dev) {
    vk_device_t* d = (vk_device_t*)dev;
    if (d) {
        vkDestroyCommandPool(d->device, d->cmd_pool, NULL);
        vkDestroyDevice(d->device, NULL);
        free(d);
    }
}

static ace_error_t vk_device_props(void* dev, void* props) {
    vk_device_t* d = (vk_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;
    p->type = ACE_BACKEND_DEVICE_VULKAN;
    strncpy(p->name, d->props.deviceName, sizeof(p->name) - 1);
    strcpy(p->vendor, "Vulkan");
    p->total_memory = d->mem_props.memoryHeaps[0].size;
    p->max_threads = d->props.limits.maxComputeWorkGroupSize[0];
    return ACE_OK;
}

static ace_error_t vk_mem_alloc(void* dev, size_t size, void** ptr) {
    vk_device_t* d = (vk_device_t*)dev;
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    vk_buffer_t* buf = (vk_buffer_t*)calloc(1, sizeof(*buf));
    buf->dev = d;
    buf->size = size;
    
    vkCreateBuffer(d->device, &buf_info, NULL, &buf->buffer);
    
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(d->device, buf->buffer, &reqs);
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = find_memory_type(d, reqs.memoryTypeBits,
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

#ifdef SHADERC_AVAILABLE
static ace_error_t vk_kernel_compile(void* dev, const char* name, const char* src,
                                      void** kernel, char** err_msg) {
    vk_device_t* d = (vk_device_t*)dev;
    
    /* Translate to GLSL */
    ace_dtype_t dtype = ACE_DTYPE_FLOAT32;
    const char* suffix = strrchr(name, '_');
    if (suffix) {
        suffix++;
        if (strcmp(suffix, "int") == 0) dtype = ACE_DTYPE_INT32;
        else if (strcmp(suffix, "double") == 0) dtype = ACE_DTYPE_FLOAT64;
    }
    
    int n_buffers = 0, n_scalars = 0;
    char* glsl = translate_to_glsl(name, src, dtype, &n_buffers, &n_scalars);
    
    /* Compile to SPIRV */
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        g_shaderc, glsl, strlen(glsl), shaderc_compute_shader, name, "main", NULL);
    
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
    
    /* Create shader module */
    VkShaderModuleCreateInfo sm_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv
    };
    VkShaderModule shader;
    VkResult vkres = vkCreateShaderModule(d->device, &sm_info, NULL, &shader);
    shaderc_result_release(result);
    free(glsl);
    
    if (vkres != VK_SUCCESS) {
        if (err_msg) *err_msg = strdup("Failed to create shader module");
        return ACE_ERROR_COMPILE;
    }
    
    /* Create descriptor set layout */
    VkDescriptorSetLayoutBinding* bindings = (VkDescriptorSetLayoutBinding*)calloc(
        n_buffers, sizeof(VkDescriptorSetLayoutBinding));
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
    
    /* Intel GPU 需要指定 VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT */
    VkDescriptorSetLayout desc_layout;
    VkResult layout_result = vkCreateDescriptorSetLayout(d->device, &dsl_info, NULL, &desc_layout);
    if (layout_result != VK_SUCCESS) {
        fprintf(stderr, "[Vulkan] Failed to create descriptor set layout: %d\n", layout_result);
    }
    free(bindings);
    
    /* Create push constant range */
    VkPushConstantRange pc_range = {0};
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &desc_layout
    };
    
    if (n_scalars > 0) {
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = n_scalars * sizeof(float);  /* All scalars as float */
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
    }
    
    VkPipelineLayout layout;
    vkCreatePipelineLayout(d->device, &pl_info, NULL, &layout);
    
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
    vkCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline);
    vkDestroyShaderModule(d->device, shader, NULL);
    
    /* Create descriptor pool - one entry per binding */
    VkDescriptorPoolSize* pool_sizes = (VkDescriptorPoolSize*)calloc(n_buffers, sizeof(VkDescriptorPoolSize));
    for (int i = 0; i < n_buffers; i++) {
        pool_sizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[i].descriptorCount = 1;
    }
    
    VkDescriptorPoolCreateInfo dp_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = n_buffers > 0 ? n_buffers : 0,
        .pPoolSizes = pool_sizes
    };
    VkDescriptorPool desc_pool;
    vkCreateDescriptorPool(d->device, &dp_info, NULL, &desc_pool);
    free(pool_sizes);
    
    /* Allocate descriptor set */
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (n_buffers > 0) {
        VkDescriptorSetAllocateInfo ds_alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &desc_layout
        };
        vkAllocateDescriptorSets(d->device, &ds_alloc, &desc_set);
    }
    
    /* Create kernel */
    vk_kernel_t* k = (vk_kernel_t*)calloc(1, sizeof(*k));
    k->pipeline = pipeline;
    k->layout = layout;
    k->desc_layout = desc_layout;
    k->desc_pool = desc_pool;
    k->desc_set = desc_set;
    k->n_buffers = n_buffers;
    k->n_scalars = n_scalars;
    k->name = strdup(name);
    k->dev = d;
    
    *kernel = k;
    return ACE_OK;
}
#else
static ace_error_t vk_kernel_compile(void* dev, const char* name, const char* src,
                                      void** kernel, char** err_msg) {
    if (err_msg) *err_msg = strdup("Vulkan requires shaderc for SPIRV compilation");
    return ACE_ERROR_COMPILE;
}
#endif

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
    vk_kernel_t* k = (vk_kernel_t*)kernel;
    vk_device_t* d = k->dev;

    /* Update descriptor sets */
    int buf_idx = 0;
    if (k->n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        VkWriteDescriptorSet* writes = (VkWriteDescriptorSet*)calloc(k->n_buffers, sizeof(VkWriteDescriptorSet));
        VkDescriptorBufferInfo* buf_infos = (VkDescriptorBufferInfo*)calloc(k->n_buffers, sizeof(VkDescriptorBufferInfo));

        for (int i = 0; i < n && buf_idx < k->n_buffers; i++) {
            if (sizes[i] == ACE_ARG_BUFFER) {
                vk_buffer_t* buf = (vk_buffer_t*)args[i];
                buf_infos[buf_idx].buffer = buf->buffer;
                buf_infos[buf_idx].offset = 0;
                buf_infos[buf_idx].range = VK_WHOLE_SIZE;  /* Use whole buffer */

                writes[buf_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[buf_idx].dstSet = k->desc_set;
                writes[buf_idx].dstBinding = buf_idx;
                writes[buf_idx].descriptorCount = 1;
                writes[buf_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[buf_idx].pBufferInfo = &buf_infos[buf_idx];
                buf_idx++;
            }
        }
        vkUpdateDescriptorSets(d->device, buf_idx, writes, 0, NULL);
        free(writes);
        free(buf_infos);
    }
    
    /* Collect scalar values for push constants */
    float* scalars = (float*)calloc(k->n_scalars > 0 ? k->n_scalars : 1, sizeof(float));
    int scalar_idx = 0;
    for (int i = 0; i < n && scalar_idx < k->n_scalars; i++) {
        if (sizes[i] == ACE_ARG_VALUE) {
            /* Read both int and float interpretations */
            int ival = *(int*)args[i];
            float fval = *(float*)args[i];
            
            /* Decide which interpretation to use:
             * If float value is reasonable (not too large/small) and int interpretation
             * looks like a large number (likely float bits), use float.
             * Otherwise use int converted to float.
             */
            if (fabsf(fval) > 1e-6f && fabsf(fval) < 1e6f && 
                (fabsf(fval) < 1.0f || fabsf(fval) - floorf(fabsf(fval)) > 1e-6f)) {
                /* Looks like a proper float with fractional part */
                scalars[scalar_idx] = fval;
            } else if (ival >= -1000000 && ival <= 1000000) {
                /* Looks like a reasonable int */
                scalars[scalar_idx] = (float)ival;
            } else {
                /* Use float interpretation */
                scalars[scalar_idx] = fval;
            }
            scalar_idx++;
        }
    }
    
    /* Create command buffer */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = d->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(d->device, &cmd_alloc, &cmd);
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &begin_info);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    
    if (k->n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &k->desc_set, 0, NULL);
    }
    
    /* Push constants */
    if (k->n_scalars > 0) {
        vkCmdPushConstants(cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 
                           k->n_scalars * sizeof(float), scalars);
    }
    
    uint32_t groups = (uint32_t)((cfg->grid[0] * cfg->block[0] + 255) / 256);
    vkCmdDispatch(cmd, groups, 1, 1);
    
    vkEndCommandBuffer(cmd);
    
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };
    vkQueueSubmit(d->queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(d->queue);
    
    vkFreeCommandBuffers(d->device, d->cmd_pool, 1, &cmd);
    free(scalars);
    return ACE_OK;
}

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

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_VULKAN, "Vulkan", &vk_ops)

#else
static ace_backend_ops_t vk_ops = {0};
ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_VULKAN, "Vulkan (unavailable)", &vk_ops)
#endif
