/**
 * @file vulkan_device_features.c
 * @brief Vulkan 设备特性检测
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdio.h>
#include <string.h>

/* 全局设备特性变量 */
vk_device_features_t g_device_features = {0};

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
void vk_detect_device_features(VkPhysicalDevice physical_device) {
    /* 初始化特性为 0 */
    memset(&g_device_features, 0, sizeof(g_device_features));

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
    g_device_features.has_float16 = float16_int8.shaderFloat16;
    g_device_features.has_int8 = float16_int8.shaderInt8;
    g_device_features.has_int16 = features2.features.shaderInt16;
    g_device_features.has_16bit_storage = storage16.storageBuffer16BitAccess;
    g_device_features.has_8bit_storage = storage8.storageBuffer8BitAccess;
    
    /* 64 位类型支持检测 - 使用 Vulkan 核心特性 */
    g_device_features.has_float64 = features2.features.shaderFloat64;
    g_device_features.has_int64 = features2.features.shaderInt64;
    /* 64 位存储支持 - 某些实现可能在 storageBuffer16BitAccess 中报告 */
    /* 保守起见，只有当明确支持 64 位操作时才认为支持 */
    g_device_features.has_64bit_storage = g_device_features.has_float64 || g_device_features.has_int64;
    
    /* BF16 扩展可能不可用，使用条件编译 */
#ifdef VK_EXT_shader_bfloat16
    g_device_features.has_bfloat16 = bfloat16.shaderBFloat16;
#else
    g_device_features.has_bfloat16 = 0;
#endif

    printf("[Vulkan] Device features detected:\n");
    printf("  - shaderFloat16: %d\n", g_device_features.has_float16);
    printf("  - shaderInt8: %d\n", g_device_features.has_int8);
    printf("  - shaderInt16: %d\n", g_device_features.has_int16);
    printf("  - storageBuffer16BitAccess: %d\n", g_device_features.has_16bit_storage);
    printf("  - storageBuffer8BitAccess: %d\n", g_device_features.has_8bit_storage);
    printf("  - shaderFloat64: %d\n", g_device_features.has_float64);
    printf("  - shaderInt64: %d\n", g_device_features.has_int64);
    printf("  - storageBuffer64BitAccess: %d\n", g_device_features.has_64bit_storage);
    printf("  - shaderBFloat16: %d\n", g_device_features.has_bfloat16);
}

/**
 * @brief 查询数据类型是否原生支持存储
 *
 * @param dtype 数据类型
 * @return 1=原生支持，0=需要模拟
 */
int vk_supports_native_storage(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:
        case ACE_DTYPE_INT32:
            return 1;  /* 基础类型总是支持 */

        case ACE_DTYPE_FLOAT64:
            /* FLOAT64 需要 shaderFloat64 + 64bit storage */
            return g_device_features.has_float64 && g_device_features.has_64bit_storage;

        case ACE_DTYPE_INT64:
            /* INT64 需要 shaderInt64 + 64bit storage */
            return g_device_features.has_int64 && g_device_features.has_64bit_storage;

        case ACE_DTYPE_FLOAT16:
            /* FP16 需要 shaderFloat16 + 16bit storage */
            return g_device_features.has_float16 && g_device_features.has_16bit_storage;

        case ACE_DTYPE_BFLOAT16:
            /* BF16 需要 shaderBFloat16 + 16bit storage，否则模拟 */
            return g_device_features.has_bfloat16 && g_device_features.has_16bit_storage;

        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
            /* INT8 需要 shaderInt8 + 8bit storage */
            return g_device_features.has_int8 && g_device_features.has_8bit_storage;

        case ACE_DTYPE_INT16:
            /* INT16 需要 shaderInt16 + 16bit storage */
            return g_device_features.has_int16 && g_device_features.has_16bit_storage;

        default:
            return 1;
    }
}

#endif /* VULKAN_AVAILABLE */
