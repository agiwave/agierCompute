/**
 * @file engine_internal.h
 * @brief AgierCompute engine internal header
 */
#ifndef ENGINE_INTERNAL_H
#define ENGINE_INTERNAL_H

#include "ace.h"
#include "ace_backend_api.h"

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
 * Internal structures
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

/* Kernel template */
typedef struct {
    char* name;
    char* src;
} kernel_template_t;

#define MAX_TEMPLATES 256
#define MAX_BACKENDS 16

/* Global engine state */
typedef struct {
    backend_entry_t list[MAX_BACKENDS];
    int count;
    int inited;
    int auto_init_attempted;
} engine_state_t;

extern engine_state_t g_engine;

extern kernel_template_t g_templates[MAX_TEMPLATES];
extern int g_template_count;

/* ============================================================================
 * Backend loading (engine_backend.c)
 * ============================================================================ */

void engine_load_backend(const char* path);
void engine_scan_dir(const char* dir);
void engine_auto_init(void);
backend_entry_t* engine_find_backend(ace_device_type_t type);

/* ============================================================================
 * Device management (engine_device.c)
 * ============================================================================ */

/* ============================================================================
 * Memory management (engine_memory.c)
 * ============================================================================ */

/* ============================================================================
 * Kernel management (engine_kernel.c)
 * ============================================================================ */

kernel_template_t* engine_get_template(ace_kernel_t kernel);

#endif /* ENGINE_INTERNAL_H */
