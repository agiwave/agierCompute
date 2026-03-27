/**
 * @file vulkan_kernel.c
 * @brief Vulkan backend kernel compilation and execution
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SHADERC_AVAILABLE

static ace_error_t compile_spirv(const char* glsl_src, uint32_t** spirv_out, size_t* spirv_size_out) {
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    if (!compiler) {
        return ACE_ERROR_COMPILE;
    }

    /* 直接使用 NULL 选项，与原始代码一致 */
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler,
        glsl_src, strlen(glsl_src),
        shaderc_compute_shader,
        "main",
        "main",
        NULL
    );

    shaderc_compiler_release(compiler);

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        printf("[Vulkan] Shader compilation error:\n%s\n", shaderc_result_get_error_message(result));
        shaderc_result_release(result);
        return ACE_ERROR_COMPILE;
    }

    *spirv_size_out = shaderc_result_get_length(result);
    *spirv_out = (uint32_t*)malloc(*spirv_size_out);
    memcpy(*spirv_out, shaderc_result_get_bytes(result), *spirv_size_out);

    shaderc_result_release(result);
    return ACE_OK;
}

#endif /* SHADERC_AVAILABLE */

static ace_error_t create_pipeline(vk_device_internal_t* d, ace_kernel_def_t* kernel_def,
                                    int* n_buffers, int* n_scalars) {
    if (d->kernel_count >= MAX_CACHED_KERNELS) return ACE_ERROR_COMPILE;

    vk_cached_kernel_t* k = &d->kernels[d->kernel_count];
    k->id = kernel_def->id * 16 + kernel_def->dtype;
    k->name = strdup(kernel_def->name);
    k->src = strdup(kernel_def->src);

    /* 翻译 GLSL 代码 */
    ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;
    char* glsl_src = vk_translate_to_glsl(kernel_def->name, kernel_def->src, dtype, n_buffers, n_scalars);
    k->n_buffers = *n_buffers;
    k->n_scalars = *n_scalars;

    /* 调试输出 - 打印生成的 GLSL 代码 */
    printf("[Vulkan] Generated GLSL for %s (dtype=%d):\n---\n%s\n---\n", 
           kernel_def->name, dtype, glsl_src);

    /* 编译 SPIR-V */
    uint32_t* spirv = NULL;
    size_t spirv_size = 0;

#ifdef SHADERC_AVAILABLE
    ace_error_t err = compile_spirv(glsl_src, &spirv, &spirv_size);
    if (err != ACE_OK) {
        free(glsl_src);
        return ACE_ERROR_COMPILE;
    }

    /* 保存 SPIR-V 到文件用于调试 */
    /*
    FILE* f = fopen("/tmp/test.spv", "wb");
    if (f) {
        fwrite(spirv, 1, spirv_size, f);
        fclose(f);
    }
    */
#else
    (void)spirv;
    (void)spirv_size;
    free(glsl_src);
    return ACE_ERROR_COMPILE;
#endif

    /* 验证 SPIR-V 魔数 */
    if (spirv_size < 4 || spirv[0] != 0x07230203) {
        printf("[Vulkan] Invalid SPIR-V magic number: 0x%08x\n", spirv[0]);
        free(spirv);
        free(glsl_src);
        return ACE_ERROR_COMPILE;
    }

    /* 创建 Shader Module */
    VkShaderModuleCreateInfo sm_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv
    };

    VkResult result = vkCreateShaderModule(d->dev->device, &sm_info, NULL, &k->shader);
    if (result != VK_SUCCESS) {
        free(spirv);
        free(glsl_src);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    /* 现在可以 free spirv 和 glsl */
    free(spirv);
    free(glsl_src);

    /* 创建 Descriptor Set Layout */
    VkDescriptorSetLayoutBinding bindings[8];
    int binding_count = 0;
    for (int i = 0; i < *n_buffers && i < 8; i++) {
        bindings[binding_count].binding = i;
        bindings[binding_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding_count].descriptorCount = 1;
        bindings[binding_count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[binding_count].pImmutableSamplers = NULL;
        binding_count++;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindings = bindings
    };

    result = vkCreateDescriptorSetLayout(d->dev->device, &dsl_info, NULL, &k->desc_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(d->dev->device, k->shader, NULL);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    /* 创建 Pipeline Layout */
    VkPushConstantRange pc_range = {0};
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k->desc_layout
    };

    if (*n_scalars > 0) {
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = (*n_scalars) * sizeof(int);
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
    }

    result = vkCreatePipelineLayout(d->dev->device, &pl_info, NULL, &k->layout);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(d->dev->device, k->desc_layout, NULL);
        vkDestroyShaderModule(d->dev->device, k->shader, NULL);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    /* 创建 Compute Pipeline */
    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = k->shader,
        .pName = "main"
    };

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_info,
        .layout = k->layout
    };

    result = vkCreateComputePipelines(d->dev->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &k->pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(d->dev->device, k->layout, NULL);
        vkDestroyDescriptorSetLayout(d->dev->device, k->desc_layout, NULL);
        vkDestroyShaderModule(d->dev->device, k->shader, NULL);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    /* 创建 Descriptor Pool 和 Set */
    VkDescriptorPoolSize pool_sizes[8];
    int pool_size_count = 0;
    for (int i = 0; i < binding_count; i++) {
        pool_sizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[pool_size_count].descriptorCount = 1;
        pool_size_count++;
    }

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = pool_size_count,
        .pPoolSizes = pool_sizes
    };

    result = vkCreateDescriptorPool(d->dev->device, &pool_info, NULL, &k->desc_pool);
    if (result != VK_SUCCESS) {
        vkDestroyPipeline(d->dev->device, k->pipeline, NULL);
        vkDestroyPipelineLayout(d->dev->device, k->layout, NULL);
        vkDestroyDescriptorSetLayout(d->dev->device, k->desc_layout, NULL);
        vkDestroyShaderModule(d->dev->device, k->shader, NULL);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = k->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->desc_layout
    };

    result = vkAllocateDescriptorSets(d->dev->device, &set_info, &k->desc_set);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorPool(d->dev->device, k->desc_pool, NULL);
        vkDestroyPipeline(d->dev->device, k->pipeline, NULL);
        vkDestroyPipelineLayout(d->dev->device, k->layout, NULL);
        vkDestroyDescriptorSetLayout(d->dev->device, k->desc_layout, NULL);
        vkDestroyShaderModule(d->dev->device, k->shader, NULL);
        free(k->name);
        free(k->src);
        return ACE_ERROR_COMPILE;
    }

    d->kernel_count++;
    return ACE_OK;
}

static vk_cached_kernel_t* find_cached_kernel(vk_device_internal_t* d, int kernel_id) {
    for (int i = 0; i < d->kernel_count; i++) {
        if (d->kernels[i].id == kernel_id) {
            return &d->kernels[i];
        }
    }
    return NULL;
}

ace_error_t vk_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                              ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    if (!d || !d->dev || !d->dev->queue) return ACE_ERROR_LAUNCH;

    int kernel_id = kernel_def->id * 16 + kernel_def->dtype;
    vk_cached_kernel_t* k = find_cached_kernel(d, kernel_id);

    int n_buffers = 0, n_scalars = 0;

    /* 如果未缓存，创建 pipeline */
    if (!k) {
        ace_error_t err = create_pipeline(d, kernel_def, &n_buffers, &n_scalars);
        if (err != ACE_OK) {
            return err;
        }
        k = &d->kernels[d->kernel_count - 1];
    } else {
        n_buffers = k->n_buffers;
        n_scalars = k->n_scalars;
    }

    /* 获取数据类型 */
    ace_dtype_t dtype = (ace_dtype_t)kernel_def->dtype;

    /* 使用静态数组避免 malloc 开销 */
    VkWriteDescriptorSet writes[8];
    VkDescriptorBufferInfo buf_infos[8];
    union { int i; float f; } push_values[8];  /* push constants 值 - 支持 int 和 float */

    /* 更新 descriptor sets */
    int buf_idx = 0;
    if (n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        memset(writes, 0, sizeof(writes));

        for (int i = 0; i < n && buf_idx < n_buffers && buf_idx < 8; i++) {
            if (sizes[i] <= 0) {
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
        vkUpdateDescriptorSets(d->dev->device, buf_idx, writes, 0, NULL);
    }

    /* Push constants - 第一个参数 n 总是 int，后续参数根据内核类型 */
    int push_count = 0;
    for (int i = 0; i < n && push_count < n_scalars && push_count < 8; i++) {
        if (sizes[i] > 0) {
            /* 所有参数统一按 4 字节传递，shader 内部会正确解释 */
            push_values[push_count].i = *(int*)args[i];
            push_count++;
        }
    }

    /* 使用环形命令缓冲池 */
    int cmd_idx = d->cmd_buffer_index;
    VkCommandBuffer cmd = d->cmd_buffers[cmd_idx];
    VkFence fence = d->fences[cmd_idx];

    /* 等待之前的 fence */
    vkWaitForFences(d->dev->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(d->dev->device, 1, &fence);

    /* 重置命令缓冲 */
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);

    if (n_buffers > 0 && k->desc_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout,
            0, 1, &k->desc_set, 0, NULL);
    }

    if (push_count > 0) {
        /* 计算 push constants 大小：每个参数 4 字节对齐（Vulkan 要求） */
        size_t push_size = push_count * 4;
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

    vkQueueSubmit(d->dev->queue, 1, &submit, fence);

    /* 更新命令缓冲索引 */
    d->cmd_buffer_index = (cmd_idx + 1) % MAX_CMD_BUFFERS;

    return ACE_OK;
}

#endif /* VULKAN_AVAILABLE */
