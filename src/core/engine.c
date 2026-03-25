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

/* ============================================================================
 * 多设备管理 API - 跨 GPU 运行
 * ============================================================================ */

ace_error_t ace_device_get_all(ace_device_list_t* list) {
    if (!list) return ACE_ERROR_INVALID;

    auto_init();

    /* 扫描所有后端类型 */
    ace_device_type_t types[] = {
        ACE_DEVICE_CUDA, ACE_DEVICE_OPENCL, ACE_DEVICE_VULKAN, ACE_DEVICE_METAL, ACE_DEVICE_CPU
    };
    int num_types = sizeof(types) / sizeof(types[0]);

    /* 计算总设备数 */
    int total = 0;
    for (int i = 0; i < num_types; i++) {
        int count = 0;
        ace_device_count(types[i], &count);
        total += count;
    }

    if (total == 0) {
        list->devices = NULL;
        list->count = 0;
        list->type = ACE_DEVICE_CPU;
        return ACE_OK;
    }

    list->devices = (ace_device_t*)calloc(total, sizeof(ace_device_t));
    if (!list->devices) return ACE_ERROR_MEM;

    list->count = 0;

    /* 优先获取 GPU 设备 */
    for (int i = 0; i < num_types; i++) {
        int count = 0;
        ace_device_count(types[i], &count);
        for (int j = 0; j < count; j++) {
            ace_device_t dev;
            if (ace_device_get(types[i], j, &dev) == ACE_OK) {
                list->devices[list->count++] = dev;
            }
        }
    }

    list->type = list->count > 0 ? list->devices[0]->backend->info->type : ACE_DEVICE_CPU;
    return ACE_OK;
}

void ace_device_list_release(ace_device_list_t* list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        ace_device_release(list->devices[i]);
    }
    free(list->devices);
    list->devices = NULL;
    list->count = 0;
}

ace_error_t ace_device_select_best(ace_device_t* dev) {
    if (!dev) return ACE_ERROR_INVALID;

    ace_device_list_t list;
    ace_error_t err = ace_device_get_all(&list);
    if (err != ACE_OK || list.count == 0) {
        *dev = NULL;
        return ACE_ERROR_NOT_FOUND;
    }

    /* 选择第一个设备（优先 GPU） */
    *dev = list.devices[0];

    /* 释放其他设备 */
    for (int i = 1; i < list.count; i++) {
        ace_device_release(list.devices[i]);
    }
    free(list.devices);

    return ACE_OK;
}

/* ============================================================================
 * 数据并行 API - 自动跨设备分片
 * ============================================================================ */

ace_error_t ace_buffer_alloc_sharded(
    ace_device_list_t* devices,
    size_t total_size,
    ace_sharded_buffer_t* sharded
) {
    if (!devices || !sharded || devices->count == 0) return ACE_ERROR_INVALID;

    int count = devices->count;
    sharded->buffers = (ace_buffer_t*)calloc(count, sizeof(ace_buffer_t));
    sharded->offsets = (size_t*)calloc(count, sizeof(size_t));
    sharded->sizes = (size_t*)calloc(count, sizeof(size_t));
    sharded->count = count;

    if (!sharded->buffers || !sharded->offsets || !sharded->sizes) {
        ace_buffer_free_sharded(sharded);
        return ACE_ERROR_MEM;
    }

    /* 平均分配 */
    size_t base_size = total_size / count;
    size_t remainder = total_size % count;

    for (int i = 0; i < count; i++) {
        sharded->sizes[i] = base_size + (i < (int)remainder ? base_size : 0);
        sharded->offsets[i] = (i == 0) ? 0 : sharded->offsets[i-1] + sharded->sizes[i-1];

        ace_error_t err = ace_buffer_alloc(devices->devices[i], sharded->sizes[i], &sharded->buffers[i]);
        if (err != ACE_OK) {
            ace_buffer_free_sharded(sharded);
            return err;
        }
    }

    return ACE_OK;
}

void ace_buffer_free_sharded(ace_sharded_buffer_t* sharded) {
    if (!sharded) return;
    for (int i = 0; i < sharded->count; i++) {
        if (sharded->buffers[i]) {
            ace_buffer_free(sharded->buffers[i]);
        }
    }
    free(sharded->buffers);
    free(sharded->offsets);
    free(sharded->sizes);
    memset(sharded, 0, sizeof(*sharded));
}

ace_error_t ace_buffer_write_sharded(
    ace_sharded_buffer_t* sharded,
    const void* data,
    size_t total_size
) {
    (void)total_size;  /* unused - sizes are in sharded->sizes */
    if (!sharded || !data) return ACE_ERROR_INVALID;

    const uint8_t* bytes = (const uint8_t*)data;
    for (int i = 0; i < sharded->count; i++) {
        ace_error_t err = ace_buffer_write(sharded->buffers[i], bytes + sharded->offsets[i], sharded->sizes[i]);
        if (err != ACE_OK) return err;
    }
    return ACE_OK;
}

ace_error_t ace_buffer_read_sharded(
    ace_sharded_buffer_t* sharded,
    void* data,
    size_t total_size
) {
    (void)total_size;  /* unused - sizes are in sharded->sizes */
    if (!sharded || !data) return ACE_ERROR_INVALID;

    uint8_t* bytes = (uint8_t*)data;
    for (int i = 0; i < sharded->count; i++) {
        ace_error_t err = ace_buffer_read(sharded->buffers[i], bytes + sharded->offsets[i], sharded->sizes[i]);
        if (err != ACE_OK) return err;
    }
    return ACE_OK;
}

ace_error_t ace_kernel_invoke_sharded(
    ace_device_list_t* devices,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
) {
    if (!devices || !kernel || devices->count == 0) return ACE_ERROR_INVALID;

    int count = devices->count;
    size_t per_device = (n + count - 1) / count;

    /* 为每个设备创建参数 */
    void** device_args[16];
    int device_types[16][16];

    for (int i = 0; i < count; i++) {
        device_args[i] = (void**)calloc(nargs, sizeof(void*));

        for (int j = 0; j < nargs; j++) {
            device_types[i][j] = types[j];
            if (types[j] == ACE_BUF) {
                /* 分片缓冲区 */
                ace_sharded_buffer_t* sharded = (ace_sharded_buffer_t*)args[j];
                device_args[i][j] = &sharded->buffers[i];
            } else {
                device_args[i][j] = args[j];
            }
        }
    }

    /* 在每个设备上启动内核 */
    for (int i = 0; i < count; i++) {
        size_t dev_n = (i == count - 1) ? (n - i * per_device) : per_device;
        if (dev_n > n) dev_n = n;

        /* 更新第一个参数（元素数量）如果是标量 */
        if (nargs > 0 && types[0] == ACE_VAL) {
            int dev_n_int = (int)dev_n;
            memcpy(device_args[i][0], &dev_n_int, sizeof(int));
        }

        ace_kernel_invoke(devices->devices[i], kernel, dtype, dev_n,
                          device_args[i], device_types[i], nargs);
    }

    /* 清理 */
    for (int i = 0; i < count; i++) {
        free(device_args[i]);
    }

    return ACE_OK;
}

ace_error_t ace_finish_all(ace_device_list_t* devices) {
    if (!devices) return ACE_ERROR_INVALID;
    for (int i = 0; i < devices->count; i++) {
        ace_finish(devices->devices[i]);
    }
    return ACE_OK;
}

/* ============================================================================
 * Stream API - 简化实现
 * ============================================================================ */

struct ace_stream_ {
    ace_device_t dev;
    int is_default;
};

ace_stream_t ace_stream_default(ace_device_t dev) {
    static struct ace_stream_ default_stream = {NULL, 1};
    if (dev) default_stream.dev = dev;
    return &default_stream;
}

ace_error_t ace_stream_create(ace_device_t dev, ace_stream_t* stream) {
    if (!dev || !stream) return ACE_ERROR_INVALID;
    *stream = (ace_stream_t)calloc(1, sizeof(struct ace_stream_));
    if (!*stream) return ACE_ERROR_MEM;
    (*stream)->dev = dev;
    (*stream)->is_default = 0;
    return ACE_OK;
}

void ace_stream_destroy(ace_stream_t stream) {
    if (stream && !stream->is_default) {
        free(stream);
    }
}

ace_error_t ace_stream_launch(
    ace_stream_t stream,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
) {
    if (!stream || !stream->dev) return ACE_ERROR_INVALID;
    /* 简化实现：直接调用同步内核执行 */
    /* 真正的异步需要后端支持 Stream */
    return ace_kernel_invoke(stream->dev, kernel, dtype, n, args, types, nargs);
}

ace_error_t ace_stream_memcpy_h2d(ace_stream_t stream, ace_buffer_t dst, const void* src, size_t size) {
    if (!stream || !dst) return ACE_ERROR_INVALID;
    return ace_buffer_write(dst, src, size);
}

ace_error_t ace_stream_memcpy_d2h(ace_stream_t stream, void* dst, ace_buffer_t src, size_t size) {
    if (!stream || !src) return ACE_ERROR_INVALID;
    return ace_buffer_read(src, dst, size);
}

ace_error_t ace_stream_synchronize(ace_stream_t stream) {
    if (!stream || !stream->dev) return ACE_ERROR_INVALID;
    return ace_finish(stream->dev);
}

/* ============================================================================
 * 内存池 API - 简化实现
 * ============================================================================ */

#define MEMPOOL_MAX_BLOCKS 256

typedef struct {
    void* ptr;
    size_t size;
    int used;
} mempool_block_t;

struct ace_mempool_ {
    ace_device_t dev;
    mempool_block_t blocks[MEMPOOL_MAX_BLOCKS];
    int count;
    size_t total_allocated;
};

ace_mempool_t ace_mempool_create(ace_device_t dev) {
    if (!dev) return NULL;
    ace_mempool_t pool = (ace_mempool_t)calloc(1, sizeof(struct ace_mempool_));
    if (pool) pool->dev = dev;
    return pool;
}

void ace_mempool_destroy(ace_mempool_t pool) {
    if (!pool) return;
    /* 释放所有未释放的块 */
    for (int i = 0; i < pool->count; i++) {
        if (pool->blocks[i].used && pool->blocks[i].ptr) {
            ace_buffer_free((ace_buffer_t)pool->blocks[i].ptr);
        }
    }
    free(pool);
}

ace_error_t ace_mempool_alloc(ace_mempool_t pool, size_t size, ace_buffer_t* buf) {
    if (!pool || !buf) return ACE_ERROR_INVALID;

    /* 查找合适的空闲块 */
    for (int i = 0; i < pool->count; i++) {
        if (!pool->blocks[i].used && pool->blocks[i].size >= size) {
            pool->blocks[i].used = 1;
            *buf = (ace_buffer_t)pool->blocks[i].ptr;
            return ACE_OK;
        }
    }

    /* 分配新块 */
    if (pool->count >= MEMPOOL_MAX_BLOCKS) return ACE_ERROR_MEM;

    ace_error_t err = ace_buffer_alloc(pool->dev, size, buf);
    if (err != ACE_OK) return err;

    pool->blocks[pool->count].ptr = *buf;
    pool->blocks[pool->count].size = size;
    pool->blocks[pool->count].used = 1;
    pool->count++;
    pool->total_allocated += size;

    return ACE_OK;
}

void ace_mempool_free(ace_mempool_t pool, ace_buffer_t buf) {
    if (!pool || !buf) return;
    /* 标记为未使用，不真正释放 */
    for (int i = 0; i < pool->count; i++) {
        if (pool->blocks[i].ptr == buf) {
            pool->blocks[i].used = 0;
            return;
        }
    }
}

/* ============================================================================
 * 实用计算原语实现
 * ============================================================================ */

ACE_KERNEL(kernel_vec_add,
    void vec_add(int n, T* a, T* b, T* y) {
        int i = GID;
        if (i < n) y[i] = a[i] + b[i];
    }
);

ACE_KERNEL(kernel_vec_scale,
    void vec_scale(int n, T alpha, T* x, T* y) {
        int i = GID;
        if (i < n) y[i] = x[i] * alpha;
    }
);

ACE_KERNEL(kernel_relu,
    void relu(int n, T* x, T* y) {
        int i = GID;
        if (i < n) y[i] = x[i] > 0 ? x[i] : 0;
    }
);

ACE_KERNEL(kernel_sigmoid,
    void sigmoid(int n, T* x, T* y) {
        int i = GID;
        if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
);

ACE_KERNEL(kernel_gemm,
    void gemm(int n, int m, int k, T* A, T* B, T* C) {
        int row = GID;
        if (row < n) {
            for (int j = 0; j < m; j++) {
                T sum = 0;
                for (int i = 0; i < k; i++) {
                    sum += A[row * k + i] * B[i * m + j];
                }
                C[row * m + j] = sum;
            }
        }
    }
);

ace_error_t ace_vec_add(ace_stream_t stream, int n, ace_buffer_t a, ace_buffer_t b, ace_buffer_t y) {
    if (!stream || !a || !b || !y) return ACE_ERROR_INVALID;
    void* args[] = {&n, a, b, y};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    return ace_stream_launch(stream, _ace_get_kernel_vec_add(), ACE_DTYPE_FLOAT32, n, args, types, 4);
}

ace_error_t ace_vec_scale(ace_stream_t stream, int n, float alpha, ace_buffer_t x, ace_buffer_t y) {
    if (!stream || !x || !y) return ACE_ERROR_INVALID;
    void* args[] = {&n, &alpha, x, y};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};
    return ace_stream_launch(stream, _ace_get_kernel_vec_scale(), ACE_DTYPE_FLOAT32, n, args, types, 4);
}

ace_error_t ace_relu(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y) {
    if (!stream || !x || !y) return ACE_ERROR_INVALID;
    void* args[] = {&n, x, y};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    return ace_stream_launch(stream, _ace_get_kernel_relu(), ACE_DTYPE_FLOAT32, n, args, types, 3);
}

ace_error_t ace_sigmoid(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y) {
    if (!stream || !x || !y) return ACE_ERROR_INVALID;
    void* args[] = {&n, x, y};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    return ace_stream_launch(stream, _ace_get_kernel_sigmoid(), ACE_DTYPE_FLOAT32, n, args, types, 3);
}

ace_error_t ace_vec_dot(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y, ace_buffer_t result) {
    /* 简化实现：使用预定义的 dot 内核 */
    /* 完整实现需要归约操作 */
    (void)stream; (void)n; (void)x; (void)y; (void)result;
    return ACE_ERROR;  /* TODO: 实现归约 */
}

ace_error_t ace_matmul(ace_stream_t stream, int m, int n, int k, ace_buffer_t A, ace_buffer_t B, ace_buffer_t C) {
    if (!stream || !A || !B || !C) return ACE_ERROR_INVALID;
    size_t sz_m = m, sz_n = n, sz_k = k;
    void* args[] = {&sz_m, &sz_n, &sz_k, A, B, C};
    int types[] = {ACE_VAL, ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    return ace_stream_launch(stream, _ace_get_kernel_gemm(), ACE_DTYPE_FLOAT32, m, args, types, 6);
}
