/**
 * @file engine.c
 * @brief AgierCompute 引擎核心 - CUDA 风格 API + JIT 编译
 */
#include "ace.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
#else
    #include <pthread.h>
    #include <unistd.h>
    #include <sys/wait.h>
#endif

/* ============================================================================
 * 内部结构
 * ============================================================================ */

typedef struct {
    ace_device_type_t type;
    int index;
    int num_threads;
} ace_device_internal_t;

typedef struct ace_kernel_internal_s {
    char* name;
    char* src;
    void* func;  /* JIT 编译后的函数指针 */
} ace_kernel_internal_t;

/* 全局当前设备 */
static struct {
    ace_device_internal_t device;
    int initialized;
} g_state = {{ACE_DEVICE_CPU, 0, 0}, 0};

/* ============================================================================
 * 线程池（CPU 后端）
 * ============================================================================ */

#define MAX_THREADS 64

typedef struct {
    void (*func)(void* args, int tid, int nthreads);
    void* args;
    int n;
    volatile int done;
#ifdef _WIN32
    HANDLE threads[MAX_THREADS];
#else
    pthread_t threads[MAX_THREADS];
#endif
    int num_threads;
} thread_job_t;

static thread_job_t g_job = {0};

#ifdef _WIN32
static DWORD WINAPI thread_worker(LPVOID param) {
    thread_job_t* job = (thread_job_t*)param;
    job->func(job->args, (int)(intptr_t)param, job->num_threads);
    return 0;
}
#else
static void* thread_worker(void* param) {
    int tid = (int)(intptr_t)param;
    thread_job_t* job = &g_job;
    job->func(job->args, tid, job->num_threads);
    return NULL;
}
#endif

static void parallel_for(void (*func)(void*, int, int), void* args, int n) {
    int num_threads = g_state.device.num_threads;
    if (num_threads <= 0) num_threads = 4;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    g_job.func = func;
    g_job.args = args;
    g_job.n = n;
    g_job.num_threads = num_threads;

#ifdef _WIN32
    for (int i = 0; i < num_threads; i++) {
        g_job.threads[i] = CreateThread(NULL, 0, thread_worker, (LPVOID)(intptr_t)i, 0, NULL);
    }
    WaitForMultipleObjects(num_threads, g_job.threads, TRUE, INFINITE);
    for (int i = 0; i < num_threads; i++) {
        CloseHandle(g_job.threads[i]);
    }
#else
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&g_job.threads[i], NULL, thread_worker, (void*)(intptr_t)i);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(g_job.threads[i], NULL);
    }
#endif
}

/* ============================================================================
 * JIT 编译器 - 使用 GCC 运行时编译
 * ============================================================================ */

static char* translate_kernel(const char* src, ace_dtype_t dtype) {
    const char* type_name = ace_dtype_str(dtype);
    
    /* 简单替换 T 为实际类型 */
    char* out = strdup(src);
    if (!out) return NULL;
    
    char* p = out;
    while ((p = strstr(p, "T")) != NULL) {
        /* 检查是否是独立的 T（不是单词的一部分） */
        int is_word = 0;
        if (p > out && (p[-1] == '_' || (p[-1] >= 'a' && p[-1] <= 'z'))) {
            is_word = 1;
        }
        if (p[1] && (p[1] == '_' || (p[1] >= 'a' && p[1] <= 'z') || (p[1] >= '0' && p[1] <= '9'))) {
            is_word = 1;
        }
        
        if (!is_word) {
            /* 替换 T */
            size_t new_len = strlen(out) + strlen(type_name);
            char* new_out = malloc(new_len);
            if (!new_out) { free(out); return NULL; }
            
            *p = '\0';
            strcpy(new_out, out);
            strcat(new_out, type_name);
            strcat(new_out, p + 1);
            free(out);
            out = new_out;
            p = new_out + strlen(new_out) - strlen(p + 1);
        } else {
            p++;
        }
    }
    
    return out;
}

static void* compile_kernel_jit(const char* name, const char* src, ace_dtype_t dtype) {
    /* 转换内核源码 */
    char* translated = translate_kernel(src, dtype);
    if (!translated) return NULL;
    
    /* 生成完整 C 代码 */
    char full_code[16384];
    snprintf(full_code, sizeof(full_code),
        "#include <math.h>\n"
        "#define GID (tid)\n"
        "#define LID (tid %% block_size)\n"
        "#define BSIZE (block_size)\n"
        "#define BARRIER() do {} while(0)\n"
        "\n"
        "typedef struct { void** args; int n; int block_size; } kernel_args_t;\n"
        "\n"
        "void %s_kernel(void* args, int tid, int nthreads) {\n"
        "    kernel_args_t* ka = (kernel_args_t*)args;\n"
        "    int block_size = ka->block_size;\n"
        "    int GID = tid;\n"
        "    int LID = tid %% block_size;\n"
        "    int BSIZE = block_size;\n",
        name
    );
    
    /* 提取内核函数体 */
    const char* body_start = strchr(translated, '{');
    const char* body_end = strrchr(translated, '}');
    if (body_start && body_end) {
        size_t body_len = body_end - body_start - 1;
        strncat(full_code, body_start + 1, body_len);
    }
    
    strcat(full_code, "\n}\n");
    free(translated);
    
    /* 写入临时文件 */
    char c_file[256], so_file[256];
    snprintf(c_file, sizeof(c_file), "/tmp/ace_%s_%d.c", name, getpid());
    snprintf(so_file, sizeof(so_file), "/tmp/ace_%s_%d.so", name, getpid());
    
    FILE* f = fopen(c_file, "w");
    if (!f) return NULL;
    fprintf(f, "%s", full_code);
    fclose(f);
    
    /* 调用 GCC 编译 */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "gcc -shared -fPIC -O2 -o %s %s -lm 2>/dev/null", so_file, c_file);
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
    void* func = GetProcAddress(lib, func_name);
#else
    #include <dlfcn.h>
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
    
    /* 存储库句柄以便后续清理 */
    /* 简化：不跟踪库句柄，程序退出时自动释放 */
    remove(so_file);
    
    return func;
}

/* ============================================================================
 * 预定义内核实现（快速路径）
 * ============================================================================ */

typedef struct {
    int n;
    void* args[16];
} kernel_ctx_t;

static void vec_add_f32(void* args, int tid, int nthreads) {
    kernel_ctx_t* ctx = (kernel_ctx_t*)args;
    int n = ctx->n;
    float* a = (float*)ctx->args[0];
    float* b = (float*)ctx->args[1];
    float* c = (float*)ctx->args[2];
    
    int chunk = (n + nthreads - 1) / nthreads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    
    for (int i = start; i < end; i++) {
        c[i] = a[i] + b[i];
    }
}

static void vec_scale_f32(void* args, int tid, int nthreads) {
    kernel_ctx_t* ctx = (kernel_ctx_t*)args;
    int n = ctx->n;
    float alpha = *(float*)ctx->args[0];
    float* in = (float*)ctx->args[1];
    float* out = (float*)ctx->args[2];
    
    int chunk = (n + nthreads - 1) / nthreads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    
    for (int i = start; i < end; i++) {
        out[i] = in[i] * alpha;
    }
}

static void relu_f32(void* args, int tid, int nthreads) {
    kernel_ctx_t* ctx = (kernel_ctx_t*)args;
    int n = ctx->n;
    float* in = (float*)ctx->args[0];
    float* out = (float*)ctx->args[1];
    
    int chunk = (n + nthreads - 1) / nthreads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    
    for (int i = start; i < end; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

static void* get_builtin_kernel(const char* name, ace_dtype_t dtype) {
    if (dtype != ACE_FLOAT32) return NULL;
    
    if (strstr(name, "vec_add")) return vec_add_f32;
    if (strstr(name, "vec_scale")) return vec_scale_f32;
    if (strstr(name, "relu")) return relu_f32;
    
    return NULL;
}

/* ============================================================================
 * 设备 API
 * ============================================================================ */

ace_error_t ace_set_device(ace_device_type_t type, int index) {
    g_state.device.type = type;
    g_state.device.index = index;
    
    if (type == ACE_DEVICE_CPU) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        g_state.device.num_threads = sysinfo.dwNumberOfProcessors;
#else
        g_state.device.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (g_state.device.num_threads <= 0) g_state.device.num_threads = 4;
    }
    
    g_state.initialized = 1;
    return ACE_OK;
}

ace_device_type_t ace_get_device(void) {
    if (!g_state.initialized) {
        ace_set_device(ACE_DEVICE_CPU, 0);
    }
    return g_state.device.type;
}

ace_error_t ace_get_device_info(ace_device_info_t* info) {
    if (!info) return ACE_ERROR;
    
    if (!g_state.initialized) {
        ace_set_device(ACE_DEVICE_CPU, 0);
    }
    
    info->type = g_state.device.type;
    info->compute_units = g_state.device.num_threads;
    info->memory = 0;
    
    switch (g_state.device.type) {
        case ACE_DEVICE_CPU:
            strcpy(info->name, "CPU");
            break;
        case ACE_DEVICE_CUDA:
            strcpy(info->name, "CUDA");
            break;
        case ACE_DEVICE_OPENCL:
            strcpy(info->name, "OpenCL");
            break;
        case ACE_DEVICE_VULKAN:
            strcpy(info->name, "Vulkan");
            break;
    }
    
    return ACE_OK;
}

ace_error_t ace_sync(void) {
    /* CPU 后端是同步的，无需额外操作 */
    return ACE_OK;
}

void ace_print_device(void) {
    ace_device_info_t info;
    ace_get_device_info(&info);
    printf("Device: %s (units: %d)\n", info.name, info.compute_units);
}

/* ============================================================================
 * 内存 API
 * ============================================================================ */

ace_error_t ace_malloc(void** ptr, size_t size) {
    *ptr = calloc(1, size);
    return *ptr ? ACE_OK : ACE_ERROR_MEM;
}

ace_error_t ace_free(void* ptr) {
    free(ptr);
    return ACE_OK;
}

ace_error_t ace_memcpy_h2d(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

ace_error_t ace_memcpy_d2h(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

ace_error_t ace_memcpy_d2d(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return ACE_OK;
}

ace_error_t ace_memcpy(void* dst, const void* src, size_t size) {
    /* 简化：假设都是主机指针 */
    memcpy(dst, src, size);
    return ACE_OK;
}

/* ============================================================================
 * 内核 API
 * ============================================================================ */

ace_kernel_t ace_kernel_register(const char* name, const char* src) {
    ace_kernel_internal_t* k = malloc(sizeof(*k));
    if (!k) return NULL;
    
    k->name = strdup(name);
    k->src = strdup(src);
    k->func = NULL;  /* JIT 编译延迟到首次调用 */
    
    return (ace_kernel_t)k;
}

ace_error_t ace_launch(ace_kernel_t kernel, size_t global_size, const char* signature, ...) {
    if (!kernel || !signature) return ACE_ERROR;
    
    ace_kernel_internal_t* k = (ace_kernel_internal_t*)kernel;
    ace_dtype_t dtype = ACE_FLOAT32;  /* 简化：默认 float32 */
    
    /* 获取或编译内核 */
    if (!k->func) {
        /* 先尝试预定义内核 */
        k->func = get_builtin_kernel(k->name, dtype);
        
        /* 如果没有，尝试 JIT 编译 */
        if (!k->func) {
            k->func = compile_kernel_jit(k->name, k->src, dtype);
        }
        
        if (!k->func) {
            fprintf(stderr, "[ACE] Failed to compile kernel: %s\n", k->name);
            return ACE_ERROR_COMPILE;
        }
    }
    
    /* 解析参数 */
    va_list args;
    va_start(args, signature);
    
    kernel_ctx_t ctx;
    ctx.n = global_size;
    
    int sig_len = strlen(signature);
    for (int i = 0; i < sig_len && i < 16; i++) {
        switch (signature[i]) {
            case 'i':  /* int */
                ctx.args[i] = malloc(sizeof(int));
                *(int*)ctx.args[i] = va_arg(args, int);
                break;
            case 'f':  /* float */
                ctx.args[i] = malloc(sizeof(float));
                *(float*)ctx.args[i] = (float)va_arg(args, double);
                break;
            case 'd':  /* double */
                ctx.args[i] = malloc(sizeof(double));
                *(double*)ctx.args[i] = va_arg(args, double);
                break;
            case 'l':  /* long */
                ctx.args[i] = malloc(sizeof(long));
                *(long*)ctx.args[i] = va_arg(args, long);
                break;
            case 'p':  /* pointer */
                ctx.args[i] = va_arg(args, void*);
                break;
            default:
                ctx.args[i] = NULL;
        }
    }
    va_end(args);
    
    /* 执行内核 */
    parallel_for(k->func, &ctx, global_size);
    
    /* 清理参数 */
    for (int i = 0; i < sig_len && i < 16; i++) {
        if (signature[i] == 'i' || signature[i] == 'f' || 
            signature[i] == 'd' || signature[i] == 'l') {
            free(ctx.args[i]);
        }
    }
    
    return ACE_OK;
}

ace_error_t ace_launch_3d(
    ace_kernel_t kernel,
    size_t grid_x, size_t grid_y, size_t grid_z,
    size_t block_x, size_t block_y, size_t block_z,
    const char* signature,
    ...
) {
    /* 简化：3D 启动转换为 1D */
    size_t total = grid_x * grid_y * grid_z * block_x * block_y * block_z;
    
    va_list args;
    va_start(args, signature);
    
    /* 重用 1D 实现 */
    kernel_ctx_t ctx;
    ctx.n = total;
    
    int sig_len = strlen(signature);
    for (int i = 0; i < sig_len && i < 16; i++) {
        switch (signature[i]) {
            case 'i':
                ctx.args[i] = malloc(sizeof(int));
                *(int*)ctx.args[i] = va_arg(args, int);
                break;
            case 'f':
                ctx.args[i] = malloc(sizeof(float));
                *(float*)ctx.args[i] = (float)va_arg(args, double);
                break;
            case 'd':
                ctx.args[i] = malloc(sizeof(double));
                *(double*)ctx.args[i] = va_arg(args, double);
                break;
            case 'l':
                ctx.args[i] = malloc(sizeof(long));
                *(long*)ctx.args[i] = va_arg(args, long);
                break;
            case 'p':
                ctx.args[i] = va_arg(args, void*);
                break;
            default:
                ctx.args[i] = NULL;
        }
    }
    va_end(args);
    
    /* 获取或编译内核 */
    ace_kernel_internal_t* k = (ace_kernel_internal_t*)kernel;
    ace_dtype_t dtype = ACE_FLOAT32;
    
    if (!k->func) {
        k->func = get_builtin_kernel(k->name, dtype);
        if (!k->func) {
            k->func = compile_kernel_jit(k->name, k->src, dtype);
        }
    }
    
    if (!k->func) {
        for (int i = 0; i < sig_len && i < 16; i++) {
            if (signature[i] == 'i' || signature[i] == 'f' || 
                signature[i] == 'd' || signature[i] == 'l') {
                free(ctx.args[i]);
            }
        }
        return ACE_ERROR_COMPILE;
    }
    
    parallel_for(k->func, &ctx, total);
    
    for (int i = 0; i < sig_len && i < 16; i++) {
        if (signature[i] == 'i' || signature[i] == 'f' || 
            signature[i] == 'd' || signature[i] == 'l') {
            free(ctx.args[i]);
        }
    }
    
    return ACE_OK;
}

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

const char* ace_strerror(ace_error_t err) {
    return ace_error_string(err);
}

const char* ace_dtype_str(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_FLOAT32: return "float";
        case ACE_FLOAT64: return "double";
        case ACE_INT32:   return "int";
        case ACE_INT64:   return "long";
        default:          return "float";
    }
}
