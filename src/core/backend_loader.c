/**
 * @file backend_loader.c
 * @brief 核心模块 - 后端动态加载器
 * 
 * 启动时扫描目录，发现并加载所有 ace_be_*.dll/so 后端模块
 */

#include "ace_backend_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
    #include <windows.h>
    #define ACE_DYNLIB_HANDLE HMODULE
    #define ACE_DYNLIB_LOAD(path) LoadLibraryA(path)
    #define ACE_DYNLIB_SYM(h, name) ((void*)GetProcAddress(h, name))
    #define ACE_DYNLIB_CLOSE(h) FreeLibrary(h)
    #define ACE_PATH_SEP '\\'
    #define ACE_BACKEND_PATTERN "ace_be_*.dll"
#else
    #include <dlfcn.h>
    #include <dirent.h>
    #define ACE_DYNLIB_HANDLE void*
    #define ACE_DYNLIB_LOAD(path) dlopen(path, RTLD_NOW)
    #define ACE_DYNLIB_SYM(h, name) dlsym(h, name)
    #define ACE_DYNLIB_CLOSE(h) dlclose(h)
    #define ACE_PATH_SEP '/'
    #define ACE_BACKEND_PATTERN "ace_be_*.so"
#endif

#define MAX_BACKENDS 16

/* 已加载的后端 */
typedef struct {
    ace_backend_info_t* info;
    ACE_DYNLIB_HANDLE handle;
    char path[512];
} loaded_backend_t;

static struct {
    loaded_backend_t backends[MAX_BACKENDS];
    int count;
    int initialized;
} g_backend_loader = {0};

/* 函数指针类型 */
typedef ace_backend_info_t* (*ace_backend_get_info_fn)(void);
typedef int (*ace_backend_api_version_fn)(void);

/**
 * 加载单个后端库
 */
static ace_error_t load_backend_lib(const char* path) {
    if (g_backend_loader.count >= MAX_BACKENDS) {
        return ACE_ERROR_OUT_OF_MEMORY;
    }
    
    ACE_DYNLIB_HANDLE handle = ACE_DYNLIB_LOAD(path);
    if (!handle) {
        return ACE_ERROR_BACKEND_LOAD_FAILED;
    }
    
    /* 获取导出函数 */
    ace_backend_get_info_fn get_info = (ace_backend_get_info_fn)
        ACE_DYNLIB_SYM(handle, "ace_backend_get_info");
    ace_backend_api_version_fn get_version = (ace_backend_api_version_fn)
        ACE_DYNLIB_SYM(handle, "ace_backend_api_version");
    
    if (!get_info || !get_version) {
        ACE_DYNLIB_CLOSE(handle);
        return ACE_ERROR_BACKEND_LOAD_FAILED;
    }
    
    /* 检查API版本 */
    if (get_version() != ACE_BACKEND_API_VERSION) {
        ACE_DYNLIB_CLOSE(handle);
        return ACE_ERROR_BACKEND_LOAD_FAILED;
    }
    
    /* 获取后端信息 */
    ace_backend_info_t* info = get_info();
    if (!info) {
        ACE_DYNLIB_CLOSE(handle);
        return ACE_ERROR_BACKEND_LOAD_FAILED;
    }
    
    /* 存储 */
    loaded_backend_t* lb = &g_backend_loader.backends[g_backend_loader.count++];
    lb->info = info;
    lb->handle = handle;
    strncpy(lb->path, path, sizeof(lb->path) - 1);
    
    return ACE_SUCCESS;
}

#ifdef _WIN32
#include <shlobj.h>

static ace_error_t scan_backends_win32(const char* dir) {
    char pattern[MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\ace_be_*.dll", dir);
    
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern, &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        return ACE_SUCCESS; /* 没找到不是错误 */
    }
    
    do {
        char fullpath[MAX_PATH];
        snprintf(fullpath, sizeof(fullpath), "%s\\%s", dir, fd.cFileName);
        load_backend_lib(fullpath);
    } while (FindNextFileA(hFind, &fd));
    
    FindClose(hFind);
    return ACE_SUCCESS;
}

#else /* Linux/macOS */

static ace_error_t scan_backends_posix(const char* dir) {
    DIR* d = opendir(dir);
    if (!d) return ACE_SUCCESS;
    
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        if (strncmp(entry->d_name, "ace_be_", 7) == 0) {
            const char* ext = strrchr(entry->d_name, '.');
            if (ext && strcmp(ext, ".so") == 0) {
                char fullpath[1024];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", dir, entry->d_name);
                load_backend_lib(fullpath);
            }
        }
    }
    closedir(d);
    return ACE_SUCCESS;
}
#endif

/**
 * 初始化后端加载器 - 扫描并加载所有后端
 */
ace_error_t ace_backend_loader_init(const char* backend_dir) {
    if (g_backend_loader.initialized) {
        return ACE_SUCCESS;
    }
    
    /* 扫描后端目录 */
    if (backend_dir) {
#ifdef _WIN32
        scan_backends_win32(backend_dir);
#else
        scan_backends_posix(backend_dir);
#endif
    }
    
    /* 初始化所有已加载的后端 */
    for (int i = 0; i < g_backend_loader.count; i++) {
        loaded_backend_t* lb = &g_backend_loader.backends[i];
        if (lb->info->ops.init && !lb->info->initialized) {
            ace_error_t err = lb->info->ops.init(lb->info);
            lb->info->initialized = (err == ACE_SUCCESS);
        }
    }
    
    g_backend_loader.initialized = 1;
    return ACE_SUCCESS;
}

/**
 * 关闭后端加载器
 */
void ace_backend_loader_shutdown(void) {
    if (!g_backend_loader.initialized) return;
    
    for (int i = 0; i < g_backend_loader.count; i++) {
        loaded_backend_t* lb = &g_backend_loader.backends[i];
        if (lb->info->ops.shutdown && lb->info->initialized) {
            lb->info->ops.shutdown(lb->info);
        }
        if (lb->handle) {
            ACE_DYNLIB_CLOSE(lb->handle);
        }
    }
    
    memset(&g_backend_loader, 0, sizeof(g_backend_loader));
}

/**
 * 获取后端数量
 */
int ace_backend_loader_count(void) {
    return g_backend_loader.count;
}

/**
 * 获取后端信息
 */
ace_backend_info_t* ace_backend_loader_get(int index) {
    if (index < 0 || index >= g_backend_loader.count) return NULL;
    return g_backend_loader.backends[index].info;
}

/**
 * 按类型查找后端
 */
ace_backend_info_t* ace_backend_loader_find(ace_device_type_t type) {
    for (int i = 0; i < g_backend_loader.count; i++) {
        if (g_backend_loader.backends[i].info->type == type) {
            return g_backend_loader.backends[i].info;
        }
    }
    return NULL;
}
