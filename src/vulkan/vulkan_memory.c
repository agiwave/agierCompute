/**
 * @file vulkan_memory.c
 * @brief Vulkan backend memory management
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdlib.h>
#include <string.h>

uint32_t vk_find_memory_type(vk_device_t* dev, uint32_t filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < dev->mem_props.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (dev->mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0;
}

ace_error_t vk_mem_alloc(void* dev, size_t size, void** ptr) {
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
        .memoryTypeIndex = vk_find_memory_type(d->dev, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    vkAllocateMemory(d->dev->device, &alloc_info, NULL, &buf->memory);
    vkBindBufferMemory(d->dev->device, buf->buffer, buf->memory, 0);

    *ptr = buf;
    return ACE_OK;
}

void vk_mem_free(void* dev, void* ptr) {
    vk_buffer_t* buf = (vk_buffer_t*)ptr;
    if (buf) {
        vkDestroyBuffer(buf->dev->device, buf->buffer, NULL);
        vkFreeMemory(buf->dev->device, buf->memory, NULL);
        free(buf);
    }
}

ace_error_t vk_mem_write(void* dev, void* dst, const void* src, size_t size) {
    vk_buffer_t* buf = (vk_buffer_t*)dst;
    void* mapped;
    vkMapMemory(buf->dev->device, buf->memory, 0, size, 0, &mapped);
    memcpy(mapped, src, size);
    vkUnmapMemory(buf->dev->device, buf->memory);
    return ACE_OK;
}

ace_error_t vk_mem_read(void* dev, void* dst, const void* src, size_t size) {
    vk_buffer_t* buf = (vk_buffer_t*)src;
    void* mapped;
    vkMapMemory(buf->dev->device, buf->memory, 0, size, 0, &mapped);
    memcpy(dst, mapped, size);
    vkUnmapMemory(buf->dev->device, buf->memory);
    return ACE_OK;
}

ace_error_t vk_finish(void* dev) {
    vk_device_internal_t* d = (vk_device_internal_t*)dev;
    if (!d || !d->dev || !d->dev->queue) return ACE_ERROR_DEVICE;
    return (vkQueueWaitIdle(d->dev->queue) == VK_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

#endif /* VULKAN_AVAILABLE */
