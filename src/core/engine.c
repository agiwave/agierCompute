/**
 * @file engine.c
 * @brief 引擎核心实现 - 后端加载和调度
 */
#include "ace.h"
#include "ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    #define DYNLIB         void*
    #define LOAD_LIB(n)    dlopen(n, RTLD_NOW)
    #define GET_SYM(h, n)  dlsym(h, n)
    #define CLOSE_LIB(h)   dlclose(h)
#endif

/* ============================================================================
 * 内部结构定义（不暴露给用户）
 * ============================================================================ */

/* 启动配置（内部使用，来自 ace_backend_api.h） */
static inline ace_launch_config_t ace_launch_1d(size_t n, size_t block) {
    ace_launch_config_t c;
    c.grid[0] = (n + block - 1) / block;
    c.grid[1] = 1; c.grid[2] = 1;
    c.block[0] = block;
    c.block[1] = 1; c.block[2] = 1;
    c.shared_mem = 0;
    return c;
}

/* 后端入口（前向声明） */
typedef struct backend_entry_s backend_entry_t;

/* 设备句柄实现 */
struct ace_device_ {
    backend_entry_t* backend;
    void* handle;
};

/* 缓冲区句柄实现 */
struct ace_buffer_ {
    ace_device_t dev;
    void* ptr;
    size_t size;
};

/* 后端入口完整定义 */
struct backend_entry_s {
    ace_backend_info_t* info;
    ace_backend_ops_t ops;
    DYNLIB handle;
    int inited;
};

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
    
    if (!get_backend) {
        get_backend = (get_backend_fn)GET_SYM(h, "ace_backend_info");
    }
    
    if (!get_backend) {
        CLOSE_LIB(h);
        return;
    }
    
    ace_backend_info_t* info = get_backend();
    if (!info) {
        CLOSE_LIB(h);
        return;
    }
    
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == info->type) {
            CLOSE_LIB(h);
            return;
        }
    }
    
    ace_backend_ops_t* ops = NULL;
    if (get_ops) {
        ops = get_ops();
    }
    
    if (!ops) {
        CLOSE_LIB(h);
        return;
    }
    
    if (ops->init && ops->init(info) != 0) {
        printf("[ACE] Backend init failed: %s\n", info->name);
        fflush(stdout);
        CLOSE_LIB(h);
        return;
    }
    
    printf("[ACE] Loaded backend: %s (type=%d)\n", info->name, info->type);
    fflush(stdout);
    
    g_engine.list[g_engine.count].info = info;
    g_engine.list[g_engine.count].ops = *ops;
    g_engine.list[g_engine.count].handle = h;
    g_engine.list[g_engine.count].inited = 1;
    g_engine.count++;
}

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
static void scan_dir(const char* dir) {
    /* POSIX: 使用 opendir/readdir */
}
#endif

/* ============================================================================
 * 自动初始化
 * ============================================================================ */

static void ace_shutdown(void);  /* 前向声明 */

static void auto_init(void) {
    if (g_engine.inited || g_engine.auto_init_attempted) return;
    
    /* 先设置标志，防止递归调用 */
    g_engine.auto_init_attempted = 1;
    
#ifdef _WIN32
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) {
        *last_slash = '\0';
        scan_dir(exe_path);
    }
#endif
    
    g_engine.inited = 1;
    
    /* 暂时禁用 atexit，避免 Vulkan 清理时崩溃 */
    /* atexit(ace_shutdown); */
}

#define ENSURE_INIT() do { if (!g_engine.inited) auto_init(); } while(0)

/* ============================================================================
 * 引擎内部初始化（已废弃，保留向后兼容）
 * ============================================================================ */

ace_error_t ace_init(const char* backend_dir) {
    ENSURE_INIT();
    return ACE_OK;
}

/* 内部清理函数（由 atexit 自动调用） */
static void ace_shutdown(void) {
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].inited && g_engine.list[i].ops.shutdown) {
            g_engine.list[i].ops.shutdown(g_engine.list[i].info);
        }
        if (g_engine.list[i].handle) {
            CLOSE_LIB(g_engine.list[i].handle);
        }
    }
    memset(&g_engine, 0, sizeof(g_engine));
}

static backend_entry_t* find_backend(ace_device_type_t type) {
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == type) {
            return &g_engine.list[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * 设备 API
 * ============================================================================ */

ace_error_t ace_device_count(ace_device_type_t type, int* count) {
    ENSURE_INIT();
    backend_entry_t* b = find_backend(type);
    if (!b || !b->ops.device_count) {
        *count = 0;
        return ACE_OK;
    }
    return b->ops.device_count(count);
}

ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev) {
    ENSURE_INIT();
    backend_entry_t* b = find_backend(type);
    if (!b || !b->ops.device_get) return ACE_ERROR_DEVICE;
    
    struct ace_device_* d = (struct ace_device_*)malloc(sizeof(*d));
    if (!d) return ACE_ERROR_MEM;
    
    d->backend = b;
    ace_error_t err = b->ops.device_get(idx, &d->handle);
    if (err != ACE_OK) {
        free(d);
        return err;
    }
    
    *dev = d;
    return ACE_OK;
}

void ace_device_release(ace_device_t dev) {
    if (dev) {
        if (dev->backend->ops.device_release) {
            dev->backend->ops.device_release(dev->handle);
        }
        free(dev);
    }
}

ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props) {
    if (!dev || !dev->backend->ops.device_props) return ACE_ERROR_DEVICE;
    return dev->backend->ops.device_props(dev->handle, props);
}

/* ============================================================================
 * 内存 API
 * ============================================================================ */

ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf) {
    if (!dev || !dev->backend->ops.mem_alloc) return ACE_ERROR_DEVICE;
    
    struct ace_buffer_* b = (struct ace_buffer_*)malloc(sizeof(*b));
    if (!b) return ACE_ERROR_MEM;
    
    b->dev = dev;
    b->size = size;
    
    ace_error_t err = dev->backend->ops.mem_alloc(dev->handle, size, &b->ptr);
    if (err != ACE_OK) {
        free(b);
        return err;
    }
    
    *buf = b;
    return ACE_OK;
}

void ace_buffer_free(ace_buffer_t buf) {
    if (buf) {
        if (buf->dev && buf->dev->backend->ops.mem_free) {
            buf->dev->backend->ops.mem_free(buf->dev->handle, buf->ptr);
        }
        free(buf);
    }
}

ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size) {
    if (!buf || !buf->dev->backend->ops.mem_write) return ACE_ERROR_DEVICE;
    return buf->dev->backend->ops.mem_write(buf->dev->handle, buf->ptr, data, size);
}

ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size) {
    if (!buf || !buf->dev->backend->ops.mem_read) return ACE_ERROR_DEVICE;
    return buf->dev->backend->ops.mem_read(buf->dev->handle, data, buf->ptr, size);
}

ace_error_t ace_finish(ace_device_t dev) {
    if (!dev || !dev->backend->ops.finish) return ACE_ERROR_DEVICE;
    return dev->backend->ops.finish(dev->handle);
}

/* ============================================================================
 * 内核模板存储
 * ============================================================================ */

#define MAX_KERNEL_TEMPLATES 128

typedef struct {
    char name[64];
    char* src;
    int in_use;
} kernel_template_t;

static kernel_template_t g_templates[MAX_KERNEL_TEMPLATES];

/* ============================================================================
 * 内核缓存（按设备+模板+类型）
 * ============================================================================ */

#define KERNEL_CACHE_SIZE 256

typedef struct {
    ace_device_t dev;
    int template_idx;
    ace_dtype_t dtype;
    void* handle;
    int in_use;
} compiled_kernel_t;

static compiled_kernel_t g_compiled[KERNEL_CACHE_SIZE];

/* ============================================================================
 * 内核注册 API
 * ============================================================================ */

ace_kernel_t ace_register_kernel(const char* name, const char* src) {
    for (int i = 0; i < MAX_KERNEL_TEMPLATES; i++) {
        if (g_templates[i].in_use && strcmp(g_templates[i].name, name) == 0) {
            return (ace_kernel_t)(intptr_t)(i + 1);
        }
    }
    
    for (int i = 0; i < MAX_KERNEL_TEMPLATES; i++) {
        if (!g_templates[i].in_use) {
            strncpy(g_templates[i].name, name, sizeof(g_templates[i].name) - 1);
            g_templates[i].src = _strdup(src);
            g_templates[i].in_use = 1;
            return (ace_kernel_t)(intptr_t)(i + 1);
        }
    }
    
    return NULL;
}

static kernel_template_t* get_template(ace_kernel_t kernel) {
    int idx = (int)(intptr_t)kernel - 1;
    if (idx < 0 || idx >= MAX_KERNEL_TEMPLATES) return NULL;
    if (!g_templates[idx].in_use) return NULL;
    return &g_templates[idx];
}

static char* instantiate_template(const char* src, ace_dtype_t dtype, char* buf, size_t size) {
    const char* type_name = ace_dtype_name(dtype);
    const char* s = src;
    char* d = buf;
    char* end = buf + size - 1;
    
    while (*s && d < end) {
        if (s[0] == 'T' && (s[1] == ' ' || s[1] == '*' || s[1] == ',' || 
                            s[1] == ')' || s[1] == ']' || s[1] == ';' ||
                            s[1] == '\n' || s[1] == '\t' || s[1] == '\0')) {
            const char* t = type_name;
            while (*t && d < end) *d++ = *t++;
            s++;
        } else {
            *d++ = *s++;
        }
    }
    *d = '\0';
    return buf;
}

static compiled_kernel_t* find_compiled_kernel(ace_device_t dev, int template_idx, ace_dtype_t dtype) {
    for (int i = 0; i < KERNEL_CACHE_SIZE; i++) {
        if (g_compiled[i].in_use && 
            g_compiled[i].dev == dev && 
            g_compiled[i].template_idx == template_idx &&
            g_compiled[i].dtype == dtype) {
            return &g_compiled[i];
        }
    }
    return NULL;
}

static compiled_kernel_t* cache_compiled_kernel(ace_device_t dev, int template_idx, 
                                                  ace_dtype_t dtype, void* handle) {
    for (int i = 0; i < KERNEL_CACHE_SIZE; i++) {
        if (!g_compiled[i].in_use) {
            g_compiled[i].dev = dev;
            g_compiled[i].template_idx = template_idx;
            g_compiled[i].dtype = dtype;
            g_compiled[i].handle = handle;
            g_compiled[i].in_use = 1;
            return &g_compiled[i];
        }
    }
    
    int idx = 0;
    if (g_compiled[idx].in_use && g_compiled[idx].handle) {
        g_compiled[idx].dev->backend->ops.kernel_release(g_compiled[idx].handle);
    }
    g_compiled[idx].dev = dev;
    g_compiled[idx].template_idx = template_idx;
    g_compiled[idx].dtype = dtype;
    g_compiled[idx].handle = handle;
    g_compiled[idx].in_use = 1;
    return &g_compiled[idx];
}

/* ============================================================================
 * 内核调用 API（lazy编译 + 执行）
 * ============================================================================ */

ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                               ace_dtype_t dtype, size_t n,
                               void** args, int* types, int nargs) {
    if (!dev || !kernel) return ACE_ERROR_DEVICE;
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
            if (err_msg) {
                printf("[ACE] Compile error: %s\n", err_msg);
                free(err_msg);
            }
            return err;
        }
        
        compiled = cache_compiled_kernel(dev, template_idx, dtype, handle);
    }
    
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
    
    ace_launch_config_t cfg = ace_launch_1d(n, 256);
    
    return dev->backend->ops.kernel_launch(compiled->handle, &cfg, processed_args, sizes, nargs);
}