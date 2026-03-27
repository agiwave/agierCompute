/**
 * @file vulkan_device_features.c
 * @brief Vulkan 设备特性检测
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdio.h>
#include <string.h>

/**
 * @brief 检测 Vulkan 物理设备的特性
 *
 * 检测内容:
 * - shaderFloat16: FP16 着色器支持
 * - shaderInt8: INT8 着色器支持
 * - shaderInt16: INT16 着色器支持
 * - storageBuffer16BitAccess: 16 位存储缓冲区支持
 * - storageBuffer8BitAccess: 8 位存储缓冲区支持
 * - shaderBFloat16: BF16 着色器支持 (VK_EXT_shader_bfloat16)
 * - shaderFloat64: FP64 着色器支持
 * - shaderInt64: INT64 着色器支持
 */
void vk_detect_device_features(VkPhysicalDevice physical_device, vk_device_features_t* features) {
    /* 初始化特性为 0 */
    memset(features, 0, sizeof(*features));

    /* 获取 Vulkan 1.1 特性 */
    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
    };

    /* Vulkan 1.2 的 16/8 位存储特性 */
    VkPhysicalDevice16BitStorageFeatures storage16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
    };

    VkPhysicalDevice8BitStorageFeaturesKHR storage8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR
    };

    /* Vulkan 1.2 的整数特性 */
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16_int8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR
    };

    /* 链接特性结构体 */
    features2.pNext = &storage16;
    storage16.pNext = &storage8;
    storage8.pNext = &float16_int8;

#ifdef VK_EXT_shader_bfloat16
    /* BF16 扩展特性 */
    VkPhysicalDeviceShaderBFloat16FeaturesKHR bfloat16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR
    };
    float16_int8.pNext = &bfloat16;
#endif

    /* 获取物理设备特性 */
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    /* 记录检测结果 */
    features->has_float16 = float16_int8.shaderFloat16;
    features->has_int8 = float16_int8.shaderInt8;
    features->has_int16 = features2.features.shaderInt16;
    features->has_16bit_storage = storage16.storageBuffer16BitAccess;
    features->has_8bit_storage = storage8.storageBuffer8BitAccess;
    
    /* 64 位类型支持检测 - 使用 Vulkan 核心特性 */
    features->has_float64 = features2.features.shaderFloat64;
    features->has_int64 = features2.features.shaderInt64;
    /* 64 位存储支持 - 某些实现可能在 storageBuffer16BitAccess 中报告 */
    /* 保守起见，只有当明确支持 64 位操作时才认为支持 */
    features->has_64bit_storage = features->has_float64 || features->has_int64;
    
    /* BF16 扩展可能不可用，使用条件编译 */
#ifdef VK_EXT_shader_bfloat16
    features->has_bfloat16 = bfloat16.shaderBFloat16;
#else
    features->has_bfloat16 = 0;
#endif

    printf("[Vulkan] Device features detected:\n");
    printf("  - shaderFloat16: %d\n", features->has_float16);
    printf("  - shaderInt8: %d\n", features->has_int8);
    printf("  - shaderInt16: %d\n", features->has_int16);
    printf("  - storageBuffer16BitAccess: %d\n", features->has_16bit_storage);
    printf("  - storageBuffer8BitAccess: %d\n", features->has_8bit_storage);
    printf("  - shaderFloat64: %d\n", features->has_float64);
    printf("  - shaderInt64: %d\n", features->has_int64);
    printf("  - storageBuffer64BitAccess: %d\n", features->has_64bit_storage);
    printf("  - shaderBFloat16: %d\n", features->has_bfloat16);
}

#endif /* VULKAN_AVAILABLE */