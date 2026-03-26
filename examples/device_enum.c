/**
 * @file device_enum.c
 * @brief 设备遍历示例
 */
#include <stdio.h>
#include "ace.h"

int main() {
    printf("========================================\n  AgierCompute - Device Enumeration\n========================================\n\n");
    
    /* 获取所有设备数量 */
    int count = 0;
    ace_device_count(ACE_DEVICE_ALL, &count);
    printf("Total devices: %d\n\n", count);
    
    /* 遍历所有设备 */
    for (int i = 0; i < count; i++) {
        ace_device_t dev = NULL;
        if (ace_device_get(ACE_DEVICE_ALL, i, &dev) == ACE_OK && dev) {
            ace_device_props_t props;
            ace_device_props(dev, &props);
            
            const char* type = props.type == ACE_DEVICE_CUDA ? "CUDA" :
                               props.type == ACE_DEVICE_OPENCL ? "OpenCL" :
                               props.type == ACE_DEVICE_VULKAN ? "Vulkan" : "CPU";
            
            printf("[%d] %s: %s\n", i, type, props.name);
            printf("    Vendor: %s, Compute units: %d, Memory: %zu MB\n",
                   props.vendor, props.compute_units, props.total_memory / (1024*1024));
            
            /* 简单测试 */
            ace_buffer_t buf;
            if (ace_buffer_alloc(dev, 1024, &buf) == ACE_OK) {
                printf("    Memory test: OK\n");
                ace_buffer_free(buf);
            }
            
            ace_device_release(dev);
            printf("\n");
        }
    }
    
    return 0;
}
