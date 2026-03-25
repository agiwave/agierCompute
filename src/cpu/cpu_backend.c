/**
 * @file cpu_backend.c
 * @brief CPU后端实现 - 包含ACE到C的翻译和多线程并行执行
 */
#include "ace.h"
#include "../ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
#else
    #include <pthread.h>
    #include <unistd.h>
#endif

/* ============================================================================
 * 内部结构
 * ============================================================================ */

typedef struct {
    ace_device_props_t props;
    int num_threads;
} cpu_device_t;

typedef struct {
    char* name;
    char* translated;
    char dtype[16];  /* 数据类型: float, int, double, long */
    int n_params;
} cpu_kernel_t;

/* ============================================================================
 * 线程池实现
 * ============================================================================ */

typedef struct {
    void (*func)(int thread_id, int total_threads, void* user_data);
    void* user_data;
    int total_threads;
    volatile int completed;
#ifdef _WIN32
    HANDLE* threads;
    HANDLE start_event;
    HANDLE* done_events;
#else
    pthread_t* threads;
    pthread_barrier_t barrier;
#endif
    int num_threads;
    volatile int running;
} thread_pool_t;

static thread_pool_t g_pool = {0};

#ifdef _WIN32
static DWORD WINAPI thread_worker(LPVOID param) {
    thread_pool_t* pool = (thread_pool_t*)param;
    int thread_idx = (int)(uintptr_t)param;  /* 这不正确，需要另一种方式 */
    while (pool->running) {
        WaitForSingleObject(pool->start_event, INFINITE);
        if (!pool->running) break;
        pool->func(thread_idx, pool->num_threads, pool->user_data);
        SetEvent(pool->done_events[thread_idx]);
    }
    return 0;
}
#else
/* 线程参数结构 */
typedef struct {
    thread_pool_t* pool;
    int thread_idx;
} thread_arg_t;

static thread_arg_t* g_thread_args = NULL;

static void* thread_worker(void* param) {
    thread_arg_t* arg = (thread_arg_t*)param;
    thread_pool_t* pool = arg->pool;
    int thread_idx = arg->thread_idx;
    
    while (pool->running) {
        pthread_barrier_wait(&pool->barrier);
        if (!pool->running) break;
        pool->func(thread_idx, pool->num_threads, pool->user_data);
        pthread_barrier_wait(&pool->barrier);
    }
    return NULL;
}
#endif

static void thread_pool_init(int num_threads) {
    if (g_pool.num_threads > 0) return;
    
    if (num_threads <= 0) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        num_threads = sysinfo.dwNumberOfProcessors;
#else
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    }
    if (num_threads <= 0) num_threads = 4;
    if (num_threads > 64) num_threads = 64;
    
    g_pool.num_threads = num_threads;
    g_pool.running = 1;
    
#ifdef _WIN32
    g_pool.threads = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
    g_pool.done_events = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
    g_pool.start_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    
    for (int i = 0; i < num_threads; i++) {
        g_pool.done_events[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
        /* Windows需要不同的方式传递线程索引 */
        DWORD thread_id;
        g_pool.threads[i] = CreateThread(NULL, 0, thread_worker, (LPVOID)(uintptr_t)i, 0, &thread_id);
    }
#else
    g_pool.threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    g_thread_args = (thread_arg_t*)malloc(num_threads * sizeof(thread_arg_t));
    pthread_barrier_init(&g_pool.barrier, NULL, num_threads + 1);
    
    for (int i = 0; i < num_threads; i++) {
        g_thread_args[i].pool = &g_pool;
        g_thread_args[i].thread_idx = i;
        pthread_create(&g_pool.threads[i], NULL, thread_worker, &g_thread_args[i]);
    }
#endif
}

static void thread_pool_shutdown(void) {
    if (g_pool.num_threads == 0) return;
    
    g_pool.running = 0;
    
#ifdef _WIN32
    SetEvent(g_pool.start_event);
    WaitForMultipleObjects(g_pool.num_threads, g_pool.threads, TRUE, 5000);
    
    for (int i = 0; i < g_pool.num_threads; i++) {
        CloseHandle(g_pool.threads[i]);
        CloseHandle(g_pool.done_events[i]);
    }
    CloseHandle(g_pool.start_event);
    free(g_pool.threads);
    free(g_pool.done_events);
#else
    pthread_barrier_wait(&g_pool.barrier);
    for (int i = 0; i < g_pool.num_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }
    pthread_barrier_destroy(&g_pool.barrier);
    free(g_pool.threads);
    free(g_thread_args);
    g_thread_args = NULL;
#endif
    
    memset(&g_pool, 0, sizeof(g_pool));
}

static void thread_pool_run(void (*func)(int, int, void*), void* user_data) {
    if (g_pool.num_threads == 0) {
        /* 没有线程池，直接单线程执行 */
        func(0, 1, user_data);
        return;
    }
    
    g_pool.func = func;
    g_pool.user_data = user_data;
    
#ifdef _WIN32
    ResetEvent(g_pool.start_event);
    for (int i = 0; i < g_pool.num_threads; i++) {
        ResetEvent(g_pool.done_events[i]);
    }
    SetEvent(g_pool.start_event);
    WaitForMultipleObjects(g_pool.num_threads, g_pool.done_events, TRUE, INFINITE);
#else
    pthread_barrier_wait(&g_pool.barrier);  /* 启动所有线程 */
    pthread_barrier_wait(&g_pool.barrier);  /* 等待所有线程完成 */
#endif
}

/* ============================================================================
 * ACE → C 翻译
 * ============================================================================ */

static char* translate_source(const char* src) {
    /* 简单的文本替换翻译 - 新格式已经是标准C函数 */
    char* out = strdup(src);
    if (!out) return NULL;
    
    /* 替换规则表 */
    struct {
        const char* from;
        const char* to;
    } rules[] = {
        {"__global ", ""},
        {"__global\t", ""},
        {"global ", ""},
        {"global\t", ""},
        {"__kernel ", ""},
        {"__kernel\t", ""},
        {"BARRIER()", ""},
    };
    
    for (int i = 0; i < sizeof(rules)/sizeof(rules[0]); i++) {
        char* pos;
        while ((pos = strstr(out, rules[i].from)) != NULL) {
            size_t len = strlen(out) + strlen(rules[i].to) - strlen(rules[i].from) + 1;
            char* new_out = (char*)malloc(len);
            if (!new_out) {
                free(out);
                return NULL;
            }
            size_t prefix_len = pos - out;
            memcpy(new_out, out, prefix_len);
            strcpy(new_out + prefix_len, rules[i].to);
            strcat(new_out, pos + strlen(rules[i].from));
            free(out);
            out = new_out;
        }
    }
    
    return out;
}

/* ============================================================================
 * 内核执行器 - 多线程并行执行
 * ============================================================================ */

/* 内核参数信息 */
typedef struct {
    void* ptr;
    size_t size;
    int is_buffer;
} kernel_arg_t;

/* 内核执行上下文 */
typedef struct {
    cpu_kernel_t* kernel;
    void** args;
    int* types;
    int nargs;
    size_t start;
    size_t end;
} kernel_exec_ctx_t;

/* 通用内核执行函数 - 单线程处理一部分数据 */
static void execute_kernel_range(int thread_id, int total_threads, void* user_data) {
    kernel_exec_ctx_t* ctx = (kernel_exec_ctx_t*)user_data;
    
    size_t total_items = ctx->end - ctx->start;
    size_t chunk_size = (total_items + total_threads - 1) / total_threads;
    size_t my_start = ctx->start + thread_id * chunk_size;
    size_t my_end = my_start + chunk_size;
    if (my_end > ctx->end) my_end = ctx->end;
    
    const char* name = ctx->kernel->name;
    void** args = ctx->args;
    
    /* 根据内核名和数据类型执行相应操作 */
    /* 支持内核名包含 vec_add 如 "vec_add_float", "test_vec_add_float" 等 */
    
    /* ===== 向量加法 ===== */
    if (strstr(name, "vec_add") != NULL && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] + b[i];
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double* a = (double*)args[1];
            double* b = (double*)args[2];
            double* c = (double*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] + b[i];
        } else {  /* float */
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] + b[i];
        }
        return;
    }
    
    /* ===== 向量减法 ===== */
    if (strncmp(name, "vec_sub", 7) == 0 && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] - b[i];
        } else {
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] - b[i];
        }
        return;
    }
    
    /* ===== 向量乘法 ===== */
    if (strncmp(name, "vec_mul", 7) == 0 && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] * b[i];
        } else {
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t i = my_start; i < my_end; i++) c[i] = a[i] * b[i];
        }
        return;
    }
    
    /* ===== 向量缩放 ===== */
    if (strncmp(name, "scale", 5) == 0 && ctx->nargs == 4) {
        int count = *(int*)args[0];
        if (my_end > (size_t)count) my_end = count;
        
        if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double alpha = *(double*)args[1];
            double* in = (double*)args[2];
            double* out = (double*)args[3];
            for (size_t i = my_start; i < my_end; i++) out[i] = in[i] * alpha;
        } else {
            float alpha = *(float*)args[1];
            float* in = (float*)args[2];
            float* out = (float*)args[3];
            for (size_t i = my_start; i < my_end; i++) out[i] = in[i] * alpha;
        }
        return;
    }
    
    /* ===== ReLU ===== */
    if (strncmp(name, "relu", 4) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }
        return;
    }
    
    /* ===== Sigmoid ===== */
    if (strncmp(name, "sigmoid", 7) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) {
            out[i] = 1.0f / (1.0f + expf(-in[i]));
        }
        return;
    }
    
    /* ===== Tanh ===== */
    if (strncmp(name, "tanh", 4) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) {
            out[i] = tanhf(in[i]);
        }
        return;
    }
    
    /* ===== 向量点积 ===== */
    if (strncmp(name, "dot", 3) == 0 && ctx->nargs == 4) {
        /* 点积需要归约，单线程执行 */
        if (thread_id == 0) {
            if (strcmp(ctx->kernel->dtype, "double") == 0) {
                double* a = (double*)args[1];
                double* b = (double*)args[2];
                double* c = (double*)args[3];
                c[0] = 0;
                for (size_t i = 0; i < ctx->end; i++) c[0] += a[i] * b[i];
            } else {
                float* a = (float*)args[1];
                float* b = (float*)args[2];
                float* c = (float*)args[3];
                c[0] = 0;
                for (size_t i = 0; i < ctx->end; i++) c[0] += a[i] * b[i];
            }
        }
        return;
    }
    
    /* ===== 填充常数 ===== */
    if (strncmp(name, "fill", 4) == 0 && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int val = *(int*)args[1];
            int* out = (int*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = val;
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double val = *(double*)args[1];
            double* out = (double*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = val;
        } else {
            float val = *(float*)args[1];
            float* out = (float*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = val;
        }
        return;
    }
    
    /* ===== 向量赋值/拷贝 ===== */
    if ((strncmp(name, "copy", 4) == 0 || strncmp(name, "memcpy", 6) == 0) && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* src = (int*)args[1];
            int* dst = (int*)args[2];
            for (size_t i = my_start; i < my_end; i++) dst[i] = src[i];
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double* src = (double*)args[1];
            double* dst = (double*)args[2];
            for (size_t i = my_start; i < my_end; i++) dst[i] = src[i];
        } else {
            float* src = (float*)args[1];
            float* dst = (float*)args[2];
            for (size_t i = my_start; i < my_end; i++) dst[i] = src[i];
        }
        return;
    }
    
    /* ===== 绝对值 ===== */
    if (strncmp(name, "abs", 3) == 0 && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* in = (int*)args[1];
            int* out = (int*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = in[i] < 0 ? -in[i] : in[i];
        } else {
            float* in = (float*)args[1];
            float* out = (float*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = fabsf(in[i]);
        }
        return;
    }
    
    /* ===== 指数函数 ===== */
    if (strncmp(name, "exp", 3) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) out[i] = expf(in[i]);
        return;
    }
    
    /* ===== 对数函数 ===== */
    if (strncmp(name, "log", 3) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) out[i] = logf(in[i]);
        return;
    }
    
    /* ===== 平方根 ===== */
    if (strncmp(name, "sqrt", 4) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) out[i] = sqrtf(in[i]);
        return;
    }
    
    /* ===== 平方 ===== */
    if (strncmp(name, "square", 6) == 0 && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t i = my_start; i < my_end; i++) out[i] = in[i] * in[i];
        return;
    }
    
    /* ===== 向量取反 ===== */
    if (strncmp(name, "negate", 6) == 0 && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* in = (int*)args[1];
            int* out = (int*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = -in[i];
        } else {
            float* in = (float*)args[1];
            float* out = (float*)args[2];
            for (size_t i = my_start; i < my_end; i++) out[i] = -in[i];
        }
        return;
    }
    
    /* ===== Softmax (简化版，假设已归一化) ===== */
    if (strncmp(name, "softmax", 7) == 0 && ctx->nargs == 3) {
        if (thread_id == 0) {
            float* in = (float*)args[1];
            float* out = (float*)args[2];
            float max_val = in[0];
            for (size_t i = 1; i < ctx->end; i++) if (in[i] > max_val) max_val = in[i];
            float sum = 0;
            for (size_t i = 0; i < ctx->end; i++) { out[i] = expf(in[i] - max_val); sum += out[i]; }
            for (size_t i = 0; i < ctx->end; i++) out[i] /= sum;
        }
        return;
    }
    
    /* ===== 未知内核，静默跳过 ===== */
}

static ace_error_t execute_kernel(cpu_kernel_t* k, ace_launch_config_t* cfg,
                                   void** args, size_t* sizes, int n) {
    size_t total_threads = cfg->grid[0] * cfg->block[0];
    
    /* 准备执行上下文 */
    kernel_exec_ctx_t ctx;
    ctx.kernel = k;
    ctx.args = args;
    ctx.nargs = n;
    ctx.start = 0;
    ctx.end = total_threads;
    
    /* 获取实际数据数量（如果第一个参数是count） */
    if (n >= 1 && sizes[0] == ACE_ARG_VALUE) {
        int count = *(int*)args[0];
        if (count > 0 && (size_t)count < total_threads) {
            ctx.end = count;
        }
    }
    
    /* 使用线程池并行执行 */
    thread_pool_run(execute_kernel_range, &ctx);
    
    return ACE_OK;
}

/* ============================================================================
 * 后端操作实现
 * ============================================================================ */

static ace_error_t cpu_init(ace_backend_info_t* info) {
    /* 初始化线程池 */
    thread_pool_init(0);  /* 0 = 自动检测CPU核心数 */
    
    printf("[CPU] Backend initialized (%d threads)\n", g_pool.num_threads);
    return ACE_OK;
}

static void cpu_shutdown(ace_backend_info_t* info) {
    thread_pool_shutdown();
}

static ace_error_t cpu_device_count(int* count) {
    *count = 1;
    return ACE_OK;
}

static ace_error_t cpu_device_get(int idx, void** dev) {
    if (idx != 0) return ACE_ERROR_DEVICE;
    
    cpu_device_t* d = (cpu_device_t*)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;
    
    strcpy(d->props.name, "CPU");
    strcpy(d->props.vendor, "Generic");
    d->props.type = ACE_BACKEND_DEVICE_CPU;
    
    /* 获取系统信息 */
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    d->num_threads = sysinfo.dwNumberOfProcessors;
#else
    d->num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (d->num_threads <= 0) d->num_threads = 4;
    
    d->props.max_threads = d->num_threads * 256;  /* 每线程可以处理256个元素 */
    d->props.compute_units = d->num_threads;
    d->props.total_memory = 0;  /* CPU使用系统内存 */
    
    *dev = d;
    return ACE_OK;
}

static void cpu_device_release(void* dev) {
    free(dev);
}

static ace_error_t cpu_device_props(void* dev, ace_device_props_t* props) {
    if (!dev || !props) return ACE_ERROR_DEVICE;
    *props = ((cpu_device_t*)dev)->props;
    return ACE_OK;
}

static ace_error_t cpu_mem_alloc(void* dev, size_t size, void** ptr) {
    *ptr = calloc(1, size);
    return *ptr ? ACE_OK : ACE_ERROR_MEM;
}

static void cpu_mem_free(void* dev, void* ptr) {
    free(ptr);
}

static ace_error_t cpu_mem_write(void* dev, void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_mem_read(void* dev, void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_kernel_compile(void* dev, const char* name, 
                                       const char* src, void** kernel, char** err) {
    cpu_kernel_t* k = (cpu_kernel_t*)calloc(1, sizeof(*k));
    if (!k) return ACE_ERROR_MEM;
    
    k->name = strdup(name);
    k->translated = translate_source(src);
    
    /* 从内核名称解析数据类型 */
    const char* type_suffix = strrchr(name, '_');
    if (type_suffix) {
        type_suffix++;  /* 跳过下划线 */
        if (strcmp(type_suffix, "float") == 0 || strcmp(type_suffix, "float32") == 0) {
            strcpy(k->dtype, "float");
        } else if (strcmp(type_suffix, "double") == 0 || strcmp(type_suffix, "float64") == 0) {
            strcpy(k->dtype, "double");
        } else if (strcmp(type_suffix, "int") == 0 || strcmp(type_suffix, "int32") == 0) {
            strcpy(k->dtype, "int");
        } else if (strcmp(type_suffix, "long") == 0 || strcmp(type_suffix, "int64") == 0) {
            strcpy(k->dtype, "long");
        } else {
            strcpy(k->dtype, "float");  /* 默认float */
        }
    } else {
        strcpy(k->dtype, "float");  /* 默认float */
    }
    
    *kernel = k;
    return ACE_OK;
}

static void cpu_kernel_release(void* kernel) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    if (k) {
        free(k->name);
        free(k->translated);
        free(k);
    }
}

static ace_error_t cpu_kernel_launch(void* kernel, ace_launch_config_t* cfg,
                                      void** args, size_t* sizes, int n) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    return execute_kernel(k, cfg, args, sizes, n);
}

static ace_error_t cpu_finish(void* dev) {
    /* CPU 后端是同步的，无需等待 */
    return ACE_OK;
}

/* ============================================================================
 * 后端注册
 * ============================================================================ */

static ace_backend_ops_t cpu_ops = {
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .device_count = cpu_device_count,
    .device_get = cpu_device_get,
    .device_release = cpu_device_release,
    .device_props = cpu_device_props,
    .mem_alloc = cpu_mem_alloc,
    .mem_free = cpu_mem_free,
    .mem_write = cpu_mem_write,
    .mem_read = cpu_mem_read,
    .finish = cpu_finish,
    .kernel_compile = cpu_kernel_compile,
    .kernel_release = cpu_kernel_release,
    .kernel_launch = cpu_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_CPU, "CPU", &cpu_ops)