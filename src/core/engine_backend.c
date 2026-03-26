/**
 * @file engine_backend.c
 * @brief Backend loading and management
 */
#include "engine_internal.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dirent.h>
#endif

/* Global engine state definition */
engine_state_t g_engine = {0};

void engine_load_backend(const char* path) {
    if (g_engine.count >= MAX_BACKENDS) return;

    DYNLIB h = LOAD_LIB(path);
    if (!h) {
#ifdef _WIN32
        printf("[ACE] Failed to load: %s (error=%lu)\n", path, GetLastError());
#else
        printf("[ACE] Failed to load: %s (%s)\n", path, dlerror());
#endif
        return;
    }

    typedef ace_backend_info_t* (*get_backend_fn)(void);
    typedef ace_backend_ops_t* (*get_ops_fn)(void);

    get_backend_fn get_backend = (get_backend_fn)GET_SYM(h, "ace_get_backend");
    get_ops_fn get_ops = (get_ops_fn)GET_SYM(h, "ace_get_backend_ops");

    if (!get_backend) get_backend = (get_backend_fn)GET_SYM(h, "ace_backend_info");
    if (!get_backend) {
        CLOSE_LIB(h);
        return;
    }

    ace_backend_info_t* info = get_backend();
    if (!info) {
        CLOSE_LIB(h);
        return;
    }

    /* 检查是否已加载 */
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == info->type) {
            CLOSE_LIB(h);
            return;
        }
    }

    ace_backend_ops_t* ops = get_ops ? get_ops() : NULL;
    if (!ops) {
        CLOSE_LIB(h);
        return;
    }

    if (ops->init) {
        ace_error_t err = ops->init(info);
        if (err != 0) {
            CLOSE_LIB(h);
            return;
        }
    }

    printf("[ACE] Loaded: %s\n", info->name);

    g_engine.list[g_engine.count].info = info;
    g_engine.list[g_engine.count].ops = *ops;
    g_engine.list[g_engine.count].handle = h;
    g_engine.list[g_engine.count].inited = 1;
    g_engine.count++;
}

#ifdef _WIN32
void engine_scan_dir(const char* dir) {
    char pattern[MAX_PATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;

    snprintf(pattern, sizeof(pattern), "%s\\ace_be_*.dll", dir);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;

    do {
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s\\%s", dir, fd.cFileName);
        engine_load_backend(path);
    } while (FindNextFileA(h, &fd));

    FindClose(h);
}
#else
void engine_scan_dir(const char* dir) {
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
                printf("[ACE] Found backend: %s\n", path);
                engine_load_backend(path);
            }
        }
    }
    closedir(d);
}
#endif

void engine_auto_init(void) {
    if (g_engine.inited || g_engine.auto_init_attempted) return;
    g_engine.auto_init_attempted = 1;

#ifdef _WIN32
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) {
        *last_slash = '\0';
        engine_scan_dir(exe_path);
    }
#else
    const char* search_dirs[] = {
        NULL, "./lib", "./bin", "../lib", "../bin", NULL
    };

    char exe_path[1024] = {0};
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char* last_slash = strrchr(exe_path, '/');
        if (last_slash) {
            *last_slash = '\0';
            search_dirs[0] = exe_path;
        }
    }

    for (int i = 0; search_dirs[i] != NULL; i++) {
        engine_scan_dir(search_dirs[i]);
    }
    engine_scan_dir(".");
#endif

    g_engine.inited = 1;
}

backend_entry_t* engine_find_backend(ace_device_type_t type) {
    for (int i = 0; i < g_engine.count; i++) {
        if (g_engine.list[i].info->type == (ace_backend_device_type_t)type) {
            return &g_engine.list[i];
        }
    }
    return NULL;
}
