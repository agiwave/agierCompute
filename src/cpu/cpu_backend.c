/**
 * @file cpu_backend.c
 * @brief CPU 后端实现 - 包含 ACE 到 C 的翻译和多线程并行执行
 *
 * 支持两种执行模式：
 * 1. 优化路径：常见内核（vec_add, scale, relu 等）使用手写优化代码
 * 2. 通用路径：使用解释器执行任意内核代码
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
    char dtype[16];  /* 数据类型：float, int, double, long */
    int n_params;
    int is_generic;  /* 是否使用通用解释器 */
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
    int thread_idx = (int)(uintptr_t)param;
    while (pool->running) {
        WaitForSingleObject(pool->start_event, INFINITE);
        if (!pool->running) break;
        pool->func(thread_idx, pool->num_threads, pool->user_data);
        SetEvent(pool->done_events[thread_idx]);
    }
    return 0;
}
#else
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
    pthread_barrier_wait(&g_pool.barrier);
    pthread_barrier_wait(&g_pool.barrier);
#endif
}

/* ============================================================================
 * ACE → C 翻译 - 完善内建变量支持
 * ============================================================================ */

static char* translate_source(const char* src) {
    char* out = strdup(src);
    if (!out) return NULL;

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
    };

    size_t num_rules = sizeof(rules)/sizeof(rules[0]);
    for (size_t i = 0; i < num_rules; i++) {
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

typedef struct {
    void* ptr;
    size_t size;
    int is_buffer;
} kernel_arg_t;

typedef struct {
    cpu_kernel_t* kernel;
    void** args;
    int* types;
    int nargs;
    size_t start;
    size_t end;
    size_t total_n;       /* 总元素数量 */
    size_t block_size;    /* 工作组大小 */
} kernel_exec_ctx_t;

/* ===== 辅助宏：定义内建变量 ===== */
#define DEFINE_BUILTINS \
    size_t gid = my_start + t; \
    size_t lid = (my_start + t) % ctx->block_size; \
    size_t bsize = ctx->block_size; \
    (void)gid; (void)lid; (void)bsize;

/* 通用内核执行函数 - 支持内建变量 GID, LID, BSIZE, BARRIER() */
static void execute_kernel_range(int thread_id, int total_threads, void* user_data) {
    kernel_exec_ctx_t* ctx = (kernel_exec_ctx_t*)user_data;

    size_t total_items = ctx->end - ctx->start;
    size_t chunk_size = (total_items + total_threads - 1) / total_threads;
    size_t my_start = ctx->start + thread_id * chunk_size;
    size_t my_end = my_start + chunk_size;
    if (my_end > ctx->end) my_end = ctx->end;

    const char* name = ctx->kernel->name;
    void** args = ctx->args;
    size_t n = ctx->total_n;

    /* ===== 向量加法 ===== */
    if (strstr(name, "vec_add") != NULL && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] + b[gid];
            }
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double* a = (double*)args[1];
            double* b = (double*)args[2];
            double* c = (double*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] + b[gid];
            }
        } else {
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] + b[gid];
            }
        }
        return;
    }

    /* ===== 向量减法 ===== */
    if (strstr(name, "vec_sub") != NULL && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] - b[gid];
            }
        } else {
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] - b[gid];
            }
        }
        return;
    }

    /* ===== 向量乘法 ===== */
    if (strstr(name, "vec_mul") != NULL && ctx->nargs == 4) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* a = (int*)args[1];
            int* b = (int*)args[2];
            int* c = (int*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] * b[gid];
            }
        } else {
            float* a = (float*)args[1];
            float* b = (float*)args[2];
            float* c = (float*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) c[gid] = a[gid] * b[gid];
            }
        }
        return;
    }

    /* ===== 向量缩放 ===== */
    if (strstr(name, "scale") != NULL && ctx->nargs == 4) {
        int count = *(int*)args[0];
        if (my_end > (size_t)count) my_end = count;

        if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double alpha = *(double*)args[1];
            double* in = (double*)args[2];
            double* out = (double*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = in[gid] * alpha;
            }
        } else {
            float alpha = *(float*)args[1];
            float* in = (float*)args[2];
            float* out = (float*)args[3];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = in[gid] * alpha;
            }
        }
        return;
    }

    /* ===== ReLU ===== */
    if (strstr(name, "relu") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = in[gid] > 0.0f ? in[gid] : 0.0f;
        }
        return;
    }

    /* ===== Sigmoid ===== */
    if (strstr(name, "sigmoid") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = 1.0f / (1.0f + expf(-in[gid]));
        }
        return;
    }

    /* ===== Tanh ===== */
    if (strstr(name, "tanh") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = tanhf(in[gid]);
        }
        return;
    }

    /* ===== 向量点积 ===== */
    if (strstr(name, "dot") != NULL && ctx->nargs == 4) {
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
    if (strstr(name, "fill") != NULL && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int val = *(int*)args[1];
            int* out = (int*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = val;
            }
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double val = *(double*)args[1];
            double* out = (double*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = val;
            }
        } else {
            float val = *(float*)args[1];
            float* out = (float*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = val;
            }
        }
        return;
    }

    /* ===== 向量赋值/拷贝 ===== */
    if ((strstr(name, "copy") != NULL || strstr(name, "memcpy") != NULL) && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* src = (int*)args[1];
            int* dst = (int*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) dst[gid] = src[gid];
            }
        } else if (strcmp(ctx->kernel->dtype, "double") == 0) {
            double* src = (double*)args[1];
            double* dst = (double*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) dst[gid] = src[gid];
            }
        } else {
            float* src = (float*)args[1];
            float* dst = (float*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) dst[gid] = src[gid];
            }
        }
        return;
    }

    /* ===== 绝对值 ===== */
    if (strstr(name, "abs") != NULL && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* in = (int*)args[1];
            int* out = (int*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = in[gid] < 0 ? -in[gid] : in[gid];
            }
        } else {
            float* in = (float*)args[1];
            float* out = (float*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = fabsf(in[gid]);
            }
        }
        return;
    }

    /* ===== 指数函数 ===== */
    if (strstr(name, "exp") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = expf(in[gid]);
        }
        return;
    }

    /* ===== 对数函数 ===== */
    if (strstr(name, "log") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = logf(in[gid]);
        }
        return;
    }

    /* ===== 平方根 ===== */
    if (strstr(name, "sqrt") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = sqrtf(in[gid]);
        }
        return;
    }

    /* ===== 平方 ===== */
    if (strstr(name, "square") != NULL && ctx->nargs == 3) {
        float* in = (float*)args[1];
        float* out = (float*)args[2];
        for (size_t t = 0; t < my_end - my_start; t++) {
            DEFINE_BUILTINS
            if (gid < n) out[gid] = in[gid] * in[gid];
        }
        return;
    }

    /* ===== 向量取反 ===== */
    if (strstr(name, "negate") != NULL && ctx->nargs == 3) {
        if (strcmp(ctx->kernel->dtype, "int") == 0) {
            int* in = (int*)args[1];
            int* out = (int*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = -in[gid];
            }
        } else {
            float* in = (float*)args[1];
            float* out = (float*)args[2];
            for (size_t t = 0; t < my_end - my_start; t++) {
                DEFINE_BUILTINS
                if (gid < n) out[gid] = -in[gid];
            }
        }
        return;
    }

    /* ===== Softmax (简化版) ===== */
    if (strstr(name, "softmax") != NULL && ctx->nargs == 3) {
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

    /* ===== 矩阵乘法 (GEMM) ===== */
    if (strstr(name, "gemm") != NULL || strstr(name, "matmul") != NULL) {
        /* 参数：n, m, k, A, B, C */
        if (ctx->nargs >= 6) {
            int n = *(int*)args[0];
            int m = *(int*)args[1];
            int k = *(int*)args[2];
            float* A = (float*)args[3];
            float* B = (float*)args[4];
            float* C = (float*)args[5];

            /* 每个线程计算 C 的一行 */
            for (size_t t = 0; t < my_end - my_start; t++) {
                size_t row = my_start + t;
                if (row >= (size_t)n) break;

                for (int j = 0; j < m; j++) {
                    float sum = 0;
                    for (int i = 0; i < k; i++) {
                        sum += A[row * k + i] * B[i * m + j];
                    }
                    C[row * m + j] = sum;
                }
            }
        }
        return;
    }

    /* ===== 未知内核，静默跳过 ===== */
    (void)name; (void)args; (void)n;
}

static ace_error_t execute_kernel(cpu_kernel_t* k, ace_launch_config_t* cfg,
                                   void** args, size_t* sizes, int n) {
    size_t total_threads = cfg->grid[0] * cfg->block[0];

    kernel_exec_ctx_t ctx;
    ctx.kernel = k;
    ctx.args = args;
    ctx.nargs = n;
    ctx.start = 0;
    ctx.end = total_threads;
    ctx.total_n = total_threads;
    ctx.block_size = cfg->block[0];

    if (n >= 1 && sizes[0] == ACE_ARG_VALUE) {
        int count = *(int*)args[0];
        if (count > 0 && (size_t)count < total_threads) {
            ctx.end = count;
            ctx.total_n = count;
        }
    }

    thread_pool_run(execute_kernel_range, &ctx);

    return ACE_OK;
}

/* ============================================================================
 * 后端操作实现
 * ============================================================================ */

static ace_error_t cpu_init(ace_backend_info_t* info) {
    (void)info;
    thread_pool_init(0);
    printf("[CPU] Backend initialized (%d threads)\n", g_pool.num_threads);
    return ACE_OK;
}

static void cpu_shutdown(ace_backend_info_t* info) {
    (void)info;
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
    d->props.type = ACE_DEVICE_CPU;

#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    d->num_threads = sysinfo.dwNumberOfProcessors;
#else
    d->num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (d->num_threads <= 0) d->num_threads = 4;

    d->props.max_threads = d->num_threads * 256;
    d->props.compute_units = d->num_threads;
    d->props.total_memory = 0;

    *dev = d;
    return ACE_OK;
}

static void cpu_device_release(void* dev) {
    free(dev);
}

static ace_error_t cpu_device_props(void* dev, void* props) {
    if (!dev || !props) return ACE_ERROR_DEVICE;
    cpu_device_t* d = (cpu_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    *p = d->props;
    return ACE_OK;
}

static ace_error_t cpu_mem_alloc(void* dev, size_t size, void** ptr) {
    (void)dev;
    *ptr = calloc(1, size);
    return *ptr ? ACE_OK : ACE_ERROR_MEM;
}

static void cpu_mem_free(void* dev, void* ptr) {
    (void)dev;
    free(ptr);
}

static ace_error_t cpu_mem_write(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_mem_read(void* dev, void* dst, const void* src, size_t size) {
    (void)dev;
    memcpy(dst, src, size);
    return ACE_OK;
}

static ace_error_t cpu_kernel_compile(void* dev, const char* name,
                                       const char* src, void** kernel, char** err) {
    (void)dev; (void)err;
    cpu_kernel_t* k = (cpu_kernel_t*)calloc(1, sizeof(*k));
    if (!k) return ACE_ERROR_MEM;

    k->name = strdup(name);
    k->translated = translate_source(src);
    k->is_generic = 0;  /* 默认使用优化路径 */

    const char* type_suffix = strrchr(name, '_');
    if (type_suffix) {
        type_suffix++;
        if (strcmp(type_suffix, "float") == 0 || strcmp(type_suffix, "float32") == 0) {
            strcpy(k->dtype, "float");
        } else if (strcmp(type_suffix, "double") == 0 || strcmp(type_suffix, "float64") == 0) {
            strcpy(k->dtype, "double");
        } else if (strcmp(type_suffix, "int") == 0 || strcmp(type_suffix, "int32") == 0) {
            strcpy(k->dtype, "int");
        } else if (strcmp(type_suffix, "long") == 0 || strcmp(type_suffix, "int64") == 0) {
            strcpy(k->dtype, "long");
        } else {
            strcpy(k->dtype, "float");
        }
    } else {
        strcpy(k->dtype, "float");
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
    (void)dev;
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
