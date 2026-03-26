/**
 * @file ace_test.c
 * @brief AgierCompute 测试框架实现
 */
#include "ace_test.h"

void ace_test_print_summary(ace_test_suite_t* suite) {
    if (!suite) return;
    int total = suite->passed + suite->failed + suite->skipped;
    printf("\n========================================\n");
    printf(" Test Summary: %s\n", suite->name);
    printf("========================================\n");
    printf("  Total:   %d\n", total);
    printf("  PASS:    %d (%.1f%%)\n", suite->passed, total > 0 ? 100.0 * suite->passed / total : 0);
    printf("  FAIL:    %d\n", suite->failed);
    printf("  SKIP:    %d\n", suite->skipped);
    printf("========================================\n");
    if (suite->failed > 0) printf("RESULT: FAILED\n");
    else if (suite->passed > 0) printf("RESULT: PASSED\n");
    else printf("RESULT: SKIPPED\n");
}

void ace_test_suite_run(ace_test_suite_t* suite) {
    if (!suite || !suite->tests) return;
    
    suite->passed = suite->failed = suite->skipped = 0;
    
    int device_count = 0;
    ace_device_count(ACE_DEVICE_ALL, &device_count);
    
    if (device_count == 0) {
        printf("[%s] No devices available\n", suite->name);
        return;
    }
    
    printf("\n========================================\n");
    printf(" Test Suite: %s\n", suite->name);
    printf(" Devices: %d\n========================================\n\n", device_count);
    
    /* 打印设备列表 */
    printf("Devices:\n");
    for (int i = 0; i < device_count; i++) {
        ace_device_t dev = NULL;
        if (ace_device_get(ACE_DEVICE_ALL, i, &dev) == ACE_OK && dev) {
            ace_device_props_t props;
            ace_device_props(dev, &props);
            const char* type_str = props.type == ACE_DEVICE_CUDA ? "CUDA" :
                                   props.type == ACE_DEVICE_OPENCL ? "OpenCL" :
                                   props.type == ACE_DEVICE_VULKAN ? "Vulkan" : "CPU";
            printf("  [%d] %s: %s\n", i, type_str, props.name);
            ace_device_release(dev);
        }
    }
    printf("\n");
    
    /* 运行测试 */
    for (int t = 0; t < suite->test_count; t++) {
        ace_test_case_t* test = &suite->tests[t];
        int pass = 0, fail = 0, skip = 0;
        
        printf("--- Test: %s ---\n", test->name);
        
        for (int d = 0; d < device_count; d++) {
            ace_device_t dev = NULL;
            if (ace_device_get(ACE_DEVICE_ALL, d, &dev) != ACE_OK || !dev) {
                skip++; continue;
            }
            
            ace_device_props_t props;
            ace_device_props(dev, &props);
            const char* type_str = props.type == ACE_DEVICE_CUDA ? "CUDA" :
                                   props.type == ACE_DEVICE_OPENCL ? "OpenCL" :
                                   props.type == ACE_DEVICE_VULKAN ? "Vulkan" : "CPU";
            
            printf("  [%s:%d] %s ... ", type_str, d, props.name);
            fflush(stdout);
            
            ace_test_result_t r = test->func(dev, test->user_data);
            if (r == ACE_TEST_PASS) { printf("PASS\n"); pass++; }
            else if (r == ACE_TEST_FAIL) { printf("FAIL\n"); fail++; }
            else { printf("SKIP\n"); skip++; }
            
            ace_device_release(dev);
        }
        
        suite->passed += pass;
        suite->failed += fail;
        suite->skipped += skip;
        printf("  Summary: %d PASS, %d FAIL, %d SKIP\n\n", pass, fail, skip);
    }
    
    ace_test_print_summary(suite);
}

void ace_benchmark_save_csv(ace_benchmark_result_t* results, int count, const char* filename) {
    if (!results || count <= 0 || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "Test,Device,Type,Index,Time(ms),GFLOPS,GB/s,Pass\n");
    for (int i = 0; i < count; i++) {
        ace_benchmark_result_t* r = &results[i];
        const char* ts = r->device_type == ACE_DEVICE_CUDA ? "CUDA" :
                         r->device_type == ACE_DEVICE_OPENCL ? "OpenCL" :
                         r->device_type == ACE_DEVICE_VULKAN ? "Vulkan" : "CPU";
        fprintf(f, "%s,%s,%s,%d,%.4f,%.4f,%.4f,%d\n",
                r->test_name, r->device_name, ts, r->device_index,
                r->elapsed_ms, r->gflops, r->bandwidth_gbs, r->passed);
    }
    fclose(f);
    printf("Results saved to %s\n", filename);
}
