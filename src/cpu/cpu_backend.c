/**
 * @file cpu_backend.c
 * @brief CPU 后端实现 - 使用 GCC JIT 编译用户内核
 * 
 * 设计原则：
 * - 不预置任何内核
 * - 用户通过 ACE_KERNEL 宏定义自己的内核
 * - 运行时使用 GCC JIT 编译用户内核代码
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
    #include <dlfcn.h>
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
    char* src;
    void* func;  /* JIT 编译后的函数指针 */
} cpu_kernel_t;

/* ============================================================================
 * 线程池实现
 * ============================================================================ */

#define MAX_THREADS 64

typedef struct {
    void (*func)(void* args, int tid, int nthreads);
    void* args;
    int n;
    int num_threads;
} thread_job_t;

static thread_job_t g_job = {0};

#ifdef _WIN32
static DWORD WINAPI thread_worker(LPVOID param) {
    thread_job_t* job = &g_job;
    job->func(job->args, (int)(intptr_t)param, job->num_threads);
    return 0;
}
#else
static void* thread_worker(void* param) {
    thread_job_t* job = &g_job;
    job->func(job->args, (int)(intptr_t)param, job->num_threads);
    return NULL;
}
#endif

static void parallel_for(void (*func)(void*, int, int), void* args, int n) {
    int num_threads = g_job.num_threads;
    if (num_threads <= 0) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        num_threads = sysinfo.dwNumberOfProcessors;
#else
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (num_threads <= 0) num_threads = 4;
    }
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    g_job.func = func;
    g_job.args = args;
    g_job.n = n;
    g_job.num_threads = num_threads;

#ifdef _WIN32
    HANDLE* threads = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
    for (int i = 0; i < num_threads; i++) {
        threads[i] = CreateThread(NULL, 0, thread_worker, NULL, 0, NULL);
    }
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
    for (int i = 0; i < num_threads; i++) CloseHandle(threads[i]);
    free(threads);
#else
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_worker, NULL);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
#endif
}

/* ============================================================================
 * JIT 编译器 - 使用 GCC 编译用户内核
 * ============================================================================ */

typedef struct {
    int n;
    void* args[16];
} jit_kernel_ctx_t;

/* 生成可编译的 C 代码 */
static char* generate_c_code(const char* name, const char* src, const char* type_name) {
    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;
    
    char* p;
    while ((p = strstr(code, "T")) != NULL) {
        /* 检查是否是独立的 T */
        int is_type = 1;
        if (p > code) {
            char prev = p[-1];
            if ((prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z') ||
                (prev >= '0' && prev <= '9') || prev == '_') is_type = 0;
        }
        if (p[1]) {
            char next = p[1];
            if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') ||
                (next >= '0' && next <= '9') || next == '_') is_type = 0;
        }
        
        if (is_type) {
            size_t rest_len = strlen(p + 1);
            size_t type_len = strlen(type_name);
            char* new_code = malloc(strlen(code) + type_len);
            if (!new_code) { free(code); return NULL; }
            
            *p = '\0';
            strcpy(new_code, code);
            strcat(new_code, type_name);
            strcat(new_code, p + 1);
            free(code);
            code = new_code;
        } else {
            p++;
        }
    }
    
    /* 生成完整的 C 文件 */
    size_t len = strlen(code) + 1024;
    char* full_code = malloc(len);
    snprintf(full_code, len,
        "#include <math.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "\n"
        "typedef struct {\n"
        "    int n;\n"
        "    void* args[16];\n"
        "} jit_ctx_t;\n"
        "\n"
        "#define GID (ctx->n)\n"
        "#define LID (tid)\n"
        "#define BSIZE (nthreads)\n"
        "#define BARRIER() do {} while(0)\n"
        "\n"
        "void %s_kernel(void* user_data, int tid, int nthreads) {\n"
        "    jit_ctx_t* ctx = (jit_ctx_t*)user_data;\n"
        "    int GID_base = tid * (ctx->n / nthreads);\n"
        "    int GID_end = (tid == nthreads - 1) ? ctx->n : GID_base + (ctx->n / nthreads);\n"
        "    for (int GID = GID_base; GID < GID_end; GID++) {\n"
        "        %s\n"
        "    }\n"
        "}\n",
        name, code
    );
    
    free(code);
    return full_code;
}

/* 编译并加载内核 */
static void* compile_kernel_jit(const char* name, const char* src, const char* type_name) {
    char c_file[256], so_file[256];
    snprintf(c_file, sizeof(c_file), "/tmp/ace_%s_%d.c", name, getpid());
    snprintf(so_file, sizeof(so_file), "/tmp/ace_%s_%d.so", name, getpid());
    
    /* 生成 C 代码 */
    char* full_code = generate_c_code(name, src, type_name);
    if (!full_code) return NULL;
    
    /* 写入文件 */
    FILE* f = fopen(c_file, "w");
    if (!f) { free(full_code); return NULL; }
    fprintf(f, "%s", full_code);
    fclose(f);
    free(full_code);
    
    /* 调用 GCC 编译 */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "gcc -shared -fPIC -O2 -o \"%s\" \"%s\" -lm 2>/dev/null", so_file, c_file);
    int ret = system(cmd);
    remove(c_file);
    
    if (ret != 0) {
        remove(so_file);
        return NULL;
    }
    
    /* 动态加载 */
#ifdef _WIN32
    HMODULE lib = LoadLibraryA(so_file);
    if (!lib) { remove(so_file); return NULL; }
    
    char func_name[128];
    snprintf(func_name, sizeof(func_name), "%s_kernel", name);
    void* func = (void*)GetProcAddress(lib, func_name);
#else
    void* lib = dlopen(so_file, RTLD_NOW);
    if (!lib) { remove(so_file); return NULL; }
    
    char func_name[128];
    snprintf(func_name, sizeof(func_name), "%s_kernel", name);
    void* func = dlsym(lib, func_name);
#endif
    
    if (!func) {
#ifdef _WIN32
        FreeLibrary(lib);
#else
        dlclose(lib);
#endif
        remove(so_file);
        return NULL;
    }
    
    /* 返回包含库句柄和函数指针的结构 */
    cpu_kernel_t* k = (cpu_kernel_t*)calloc(1, sizeof(*k));
    k->name = strdup(name);
    k->src = strdup(src);
    k->func = func;
    
    remove(so_file);
    return k;
}

/* ============================================================================
 * 后端操作实现
 * ============================================================================ */

static ace_error_t cpu_init(ace_backend_info_t* info) {
    (void)info;
    printf("[CPU] Backend initialized\n");
    return ACE_OK;
}

static void cpu_shutdown(ace_backend_info_t* info) {
    (void)info;
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
                                       const char* src, void** kernel, char** err_msg) {
    (void)dev; (void)err_msg;
    
    cpu_kernel_t* k = (cpu_kernel_t*)calloc(1, sizeof(*k));
    if (!k) return ACE_ERROR_MEM;

    k->name = strdup(name);
    k->src = strdup(src);
    k->func = NULL;  /* 延迟编译到第一次执行 */

    *kernel = k;
    return ACE_OK;
}

static void cpu_kernel_release(void* kernel) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    if (k) {
        free(k->name);
        free(k->src);
        free(k);
    }
}

static ace_error_t cpu_kernel_launch(void* kernel, ace_launch_config_t* cfg,
                                      void** args, size_t* sizes, int n) {
    cpu_kernel_t* k = (cpu_kernel_t*)kernel;
    if (!k) return ACE_ERROR_LAUNCH;
    
    /* 延迟编译 */
    if (!k->func) {
        k->func = compile_kernel_jit(k->name, k->src, "float");  /* 默认 float */
        if (!k->func) return ACE_ERROR_COMPILE;
    }
    
    /* 准备执行上下文 */
    jit_kernel_ctx_t ctx;
    ctx.n = (int)(cfg->grid[0] * cfg->block[0]);
    for (int i = 0; i < n && i < 16; i++) {
        ctx.args[i] = args[i];
    }
    
    /* 并行执行 */
    parallel_for(k->func, &ctx, ctx.n);
    
    return ACE_OK;
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
