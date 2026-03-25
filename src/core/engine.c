/**
 * @file engine.c
 * @brief 引擎核心实现 - 后端加载和API转发
 * 
 * 设计原则：
 * - Core层只做API抽象和转发
 * - 异步执行、同步、内存管理由后端负责
 * - 不模拟实现Stream/Event/Mempool
 */
#include "ace.h"
#include "ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ============================================================================
 * 平台相关宏
 * ============================================================================ */
#ifdef _WIN32
    #include <windows.h>
    #define DYNLIB         HMODULE
    #define LOAD_LIB(n)    LoadLibraryA(n)
    #define GET_SYM(h, n)  ((void*)GetProcAddress(h, n))
    #define CLOSE_LIB(h)   FreeLibrary(h)
#else
    #include <dlfcn.h>
    #include <unistd.h>
    #define DYNLIB         void*
    #define LOAD_LIB(n)    dlopen(n, RTLD_NOW)
    #define GET_SYM(h, n)  dlsym(h, n)
    #define CLOSE_LIB(h)   dlclose(h)
#endif

/* ============================================================================
 * 内部结构
 * ============================================================================ */

typedef struct backend_entry_s backend_entry_t;

struct ace_device_ {
    backend_entry_t* backend;
    void* handle;
};

struct ace_buffer_ {
    ace_device_t dev;
    void* ptr;
    size_t size;
};

struct backend_entry_s {
    ace_backend_info_t* info;
    ace_backend_ops_t ops;
    DYNLIB handle;
    int inited;
};

/* 内核模板 */
typedef struct {
    char* name;
    char* src;
} kernel_template_t;

#define MAX_TEMPLATES 256
static kernel_template_t g_templates[MAX_TEMPLATES];
static int g_template_count = 0;

/* 编译后的内核缓存 */
typedef struct {
    ace_device_t dev;
    int template_idx;
    ace_dtype_t dtype;
    void* handle;
} compiled_kernel_t;

#define MAX_COMPILED 1024
static compiled_kernel_t g_compiled[MAX_COMPILED];
static int g_compiled_count = 0;

/* 全局引擎状态 */
static struct {
    backend_entry_t list[16];
    int count;
    int inited;
    int auto_init_attempted;
} g_engine;

/* ============================================================================
 * 后端加载
 * ============================================================================ */

static void load_backend(const char* path) {
    if (g_engine.count >= 16) return;
    
    DYNLIB h = LOAD_LIB(path);
    if (!h) return;
    
    typedef ace_backend_info_t* (*get_backend_fn)(void);
    typedef ace_backend_ops_t* (*get_ops_fn)(void);
    
    get_backend_fn get_backend = (get_backend_fn)GET_SYM(h, "ace_get_backend");
    get_ops_fn get_ops = (get_ops_fn)GET_SYM(h, "ace_get_backend_ops");
    
    if (!get_backend) get_backend = (get_backend_fn)GET_SYM(h, "ace_backend_info");
    if (!get_backend) { CLOSE_LIB(h); return; }
    
    ace_backend_info_t* info = get_backend();
    if (!info) { CLOSE_LIB(h); return; }
    
    /* 检查是否已加载 */
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == info->type) {
            CLOSE_LIB(h);
            return;
        }
    }
    
    ace_backend_ops_t* ops = get_ops ? get_ops() : NULL;
    if (!ops) { CLOSE_LIB(h); return; }
    
    if (ops->init && ops->init(info) != 0) {
        CLOSE_LIB(h);
        return;
    }
    
    printf("[ACE] Loaded: %s\n", info->name);
    
    g_engine.list[g_engine.count].info = info;
    g_engine.list[g_engine.count].ops = *ops;
    g_engine.list[g_engine.count].handle = h;
    g_engine.list[g_engine.count].inited = 1;
    g_engine.count++;
}

/* 扫描目录查找后端库 */
#ifdef _WIN32
static void scan_dir(const char* dir) {
    char pattern[MAX_PATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    
    snprintf(pattern, sizeof(pattern), "%s\\ace_be_*.dll", dir);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    
    do {
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s\\%s", dir, fd.cFileName);
        load_backend(path);
    } while (FindNextFileA(h, &fd));
    
    FindClose(h);
}
#else
#include <dirent.h>
static void scan_dir(const char* dir) {
    DIR* d = opendir(dir);
    if (!d) return;
    
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        const char* name = entry->d_name;
        int is_backend = 0;
        
        if (strncmp(name, "ace_be_", 7) == 0) is_backend = 1;
        else if (strncmp(name, "libace_be_", 10) == 0) is_backend = 1;
        
        if (is_backend) {
            const char* ext = strrchr(name, '.');
            if (ext && strcmp(ext, ".so") == 0) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s", dir, name);
                load_backend(path);
            }
        }
    }
    closedir(d);
}
#endif

static void auto_init(void) {
    if (g_engine.inited || g_engine.auto_init_attempted) return;
    g_engine.auto_init_attempted = 1;
    
#ifdef _WIN32
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) { *last_slash = '\0'; scan_dir(exe_path); }
#else
    const char* search_dirs[] = {
        NULL, "./lib", "./bin", "../lib", "../bin", NULL
    };
    
    char exe_path[1024] = {0};
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char* last_slash = strrchr(exe_path, '/');
        if (last_slash) { *last_slash = '\0'; search_dirs[0] = exe_path; }
    }
    
    for (int i = 0; search_dirs[i] != NULL; i++) {
        scan_dir(search_dirs[i]);
    }
    scan_dir(".");
#endif
    
    g_engine.inited = 1;
}

static backend_entry_t* find_backend(ace_device_type_t type) {
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == (ace_backend_device_type_t)type) {
            return &g_engine.list[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * 设备管理 API
 * ============================================================================ */

ace_error_t ace_device_count(ace_device_type_t type, int* count) {
    auto_init();
    
    if (type == ACE_DEVICE_CPU) {
        *count = 1;  /* CPU总是可用 */
        return ACE_OK;
    }
    
    backend_entry_t* b = find_backend(type);
    if (!b || !b->ops.device_count) {
        *count = 0;
        return ACE_OK;
    }
    
    return b->ops.device_count(count);
}

ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev) {
    auto_init();
    
    backend_entry_t* b = find_backend(type);
    if (!b) return ACE_ERROR_NOT_FOUND;
    
    ace_device_t d = (ace_device_t)calloc(1, sizeof(*d));
    if (!d) return ACE_ERROR_MEM;
    
    d->backend = b;
    
    if (b->ops.device_get) {
        ace_error_t err = b->ops.device_get(idx, &d->handle);
        if (err != ACE_OK) { free(d); return err; }
    }
    
    *dev = d;
    return ACE_OK;
}

void ace_device_release(ace_device_t dev) {
    if (!dev) return;
    if (dev->backend && dev->backend->ops.device_release) {
        dev->backend->ops.device_release(dev->handle);
    }
    free(dev);
}

ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props) {
    if (!dev || !props) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.device_props) return ACE_ERROR_BACKEND;
    return dev->backend->ops.device_props(dev->handle, props);
}

/* ============================================================================
 * 内存管理 API
 * ============================================================================ */

ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf) {
    if (!dev || !buf) return ACE_ERROR_INVALID;
    
    ace_buffer_t b = (ace_buffer_t)calloc(1, sizeof(*b));
    if (!b) return ACE_ERROR_MEM;
    
    b->dev = dev;
    b->size = size;
    
    if (dev->backend && dev->backend->ops.mem_alloc) {
        ace_error_t err = dev->backend->ops.mem_alloc(dev->handle, size, &b->ptr);
        if (err != ACE_OK) { free(b); return err; }
    }
    
    *buf = b;
    return ACE_OK;
}

void ace_buffer_free(ace_buffer_t buf) {
    if (!buf) return;
    if (buf->dev && buf->dev->backend && buf->dev->backend->ops.mem_free) {
        buf->dev->backend->ops.mem_free(buf->dev->handle, buf->ptr);
    }
    free(buf);
}

ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size) {
    if (!buf || !data || !buf->dev) return ACE_ERROR_INVALID;
    if (!buf->dev->backend || !buf->dev->backend->ops.mem_write) return ACE_ERROR_BACKEND;
    return buf->dev->backend->ops.mem_write(buf->dev->handle, buf->ptr, data, size);
}

ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size) {
    if (!buf || !data || !buf->dev) return ACE_ERROR_INVALID;
    if (!buf->dev->backend || !buf->dev->backend->ops.mem_read) return ACE_ERROR_BACKEND;
    /* 后端会自动同步 */
    return buf->dev->backend->ops.mem_read(buf->dev->handle, data, buf->ptr, size);
}

/* ============================================================================
 * 同步 API
 * ============================================================================ */

ace_error_t ace_finish(ace_device_t dev) {
    if (!dev) return ACE_ERROR_INVALID;
    if (!dev->backend || !dev->backend->ops.finish) return ACE_ERROR_BACKEND;
    return dev->backend->ops.finish(dev->handle);
}

/* ============================================================================
 * 内核模板管理
 * ============================================================================ */

static void instantiate_template(const char* src, ace_dtype_t dtype, char* out, size_t max_len) {
    const char* type_name = ace_dtype_name(dtype);
    size_t len = strlen(src);
    
    /* 替换 T 为具体类型 */
    size_t j = 0;
    for (size_t i = 0; i < len && j < max_len - 1; i++) {
        if (src[i] == 'T' && (i == 0 || !isalnum(src[i-1]))) {
            /* 检查下一个字符不是字母数字 */
            if (!isalnum(src[i+1])) {
                const char* t = type_name;
                while (*t && j < max_len - 1) out[j++] = *t++;
                continue;
            }
        }
        out[j++] = src[i];
    }
    out[j] = '\0';
}

static kernel_template_t* get_template(ace_kernel_t kernel) {
    int idx = (int)(intptr_t)kernel - 1;
    if (idx < 0 || idx >= g_template_count) return NULL;
    return &g_templates[idx];
}

static compiled_kernel_t* find_compiled_kernel(ace_device_t dev, int template_idx, ace_dtype_t dtype) {
    for (int i = 0; i < g_compiled_count; i++) {
        if (g_compiled[i].dev == dev && 
            g_compiled[i].template_idx == template_idx &&
            g_compiled[i].dtype == dtype) {
            return &g_compiled[i];
        }
    }
    return NULL;
}

static compiled_kernel_t* cache_compiled_kernel(ace_device_t dev, int template_idx, 
                                                  ace_dtype_t dtype, void* handle) {
    if (g_compiled_count >= MAX_COMPILED) return NULL;
    
    g_compiled[g_compiled_count].dev = dev;
    g_compiled[g_compiled_count].template_idx = template_idx;
    g_compiled[g_compiled_count].dtype = dtype;
    g_compiled[g_compiled_count].handle = handle;
    g_compiled_count++;
    
    return &g_compiled[g_compiled_count - 1];
}

ace_kernel_t ace_register_kernel(const char* name, const char* src) {
    if (g_template_count >= MAX_TEMPLATES) return NULL;
    
    /* 检查是否已注册 */
    for (int i = 0; i < g_template_count; i++) {
        if (strcmp(g_templates[i].name, name) == 0) {
            return (ace_kernel_t)(intptr_t)(i + 1);
        }
    }
    
    g_templates[g_template_count].name = strdup(name);
    g_templates[g_template_count].src = strdup(src);
    g_template_count++;
    
    return (ace_kernel_t)(intptr_t)g_template_count;
}

/* ============================================================================
 * 内核执行 API
 * ============================================================================ */

ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                               ace_dtype_t dtype, size_t n,
                               void** args, int* types, int nargs) {
    if (!dev || !kernel) return ACE_ERROR_INVALID;
    if (!dev->backend->ops.kernel_compile || !dev->backend->ops.kernel_launch) {
        return ACE_ERROR_BACKEND;
    }
    
    kernel_template_t* tmpl = get_template(kernel);
    if (!tmpl) return ACE_ERROR_COMPILE;
    
    int template_idx = (int)(intptr_t)kernel - 1;
    compiled_kernel_t* compiled = find_compiled_kernel(dev, template_idx, dtype);
    
    if (!compiled) {
        char instantiated[8192];
        instantiate_template(tmpl->src, dtype, instantiated, sizeof(instantiated));
        
        char full_name[128];
        snprintf(full_name, sizeof(full_name), "%s_%s", tmpl->name, ace_dtype_name(dtype));
        
        void* handle = NULL;
        char* err_msg = NULL;
        ace_error_t err = dev->backend->ops.kernel_compile(dev->handle, full_name, 
                                                            instantiated, &handle, &err_msg);
        if (err != ACE_OK) {
            if (err_msg) { printf("[ACE] Compile error: %s\n", err_msg); free(err_msg); }
            return err;
        }
        
        compiled = cache_compiled_kernel(dev, template_idx, dtype, handle);
    }
    
    /* 处理参数 */
    void* processed_args[16];
    size_t sizes[16];
    if (nargs > 16) nargs = 16;
    
    for (int i = 0; i < nargs; i++) {
        if (types[i] == ACE_BUF) {
            struct ace_buffer_* buf = (struct ace_buffer_*)args[i];
            processed_args[i] = buf ? buf->ptr : NULL;
            sizes[i] = ACE_ARG_BUFFER;
        } else {
            processed_args[i] = args[i];
            sizes[i] = ACE_ARG_VALUE;
        }
    }
    
    /* 后端负责异步执行 */
    ace_launch_config_t cfg = ace_launch_1d(n, 256);
    return dev->backend->ops.kernel_launch(compiled->handle, &cfg, processed_args, sizes, nargs);
}

ace_error_t ace_kernel_launch(ace_device_t dev, ace_kernel_t kernel,
                               ace_dtype_t dtype, ace_launch_config_t* config,
                               void** args, int* types, int nargs) {
    if (!dev || !kernel) return ACE_ERROR_INVALID;
    if (!dev->backend->ops.kernel_compile || !dev->backend->ops.kernel_launch) {
        return ACE_ERROR_BACKEND;
    }
    
    kernel_template_t* tmpl = get_template(kernel);
    if (!tmpl) return ACE_ERROR_COMPILE;
    
    int template_idx = (int)(intptr_t)kernel - 1;
    compiled_kernel_t* compiled = find_compiled_kernel(dev, template_idx, dtype);
    
    if (!compiled) {
        char instantiated[8192];
        instantiate_template(tmpl->src, dtype, instantiated, sizeof(instantiated));
        
        char full_name[128];
        snprintf(full_name, sizeof(full_name), "%s_%s", tmpl->name, ace_dtype_name(dtype));
        
        void* handle = NULL;
        char* err_msg = NULL;
        ace_error_t err = dev->backend->ops.kernel_compile(dev->handle, full_name, 
                                                            instantiated, &handle, &err_msg);
        if (err != ACE_OK) {
            if (err_msg) { printf("[ACE] Compile error: %s\n", err_msg); free(err_msg); }
            return err;
        }
        
        compiled = cache_compiled_kernel(dev, template_idx, dtype, handle);
    }
    
    /* 处理参数 */
    void* processed_args[16];
    size_t sizes[16];
    if (nargs > 16) nargs = 16;
    
    for (int i = 0; i < nargs; i++) {
        if (types[i] == ACE_BUF) {
            struct ace_buffer_* buf = (struct ace_buffer_*)args[i];
            processed_args[i] = buf ? buf->ptr : NULL;
            sizes[i] = ACE_ARG_BUFFER;
        } else {
            processed_args[i] = args[i];
            sizes[i] = ACE_ARG_VALUE;
        }
    }
    
    /* 使用自定义配置，后端负责异步执行 */
    ace_launch_config_t default_cfg = ace_launch_1d(1, 1);
    return dev->backend->ops.kernel_launch(compiled->handle,
                                            config ? config : &default_cfg,
                                            processed_args, sizes, nargs);
}
