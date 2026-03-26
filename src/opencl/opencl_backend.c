/**
 * @file opencl_backend.c
 * @brief OpenCL backend using official OpenCL SDK
 */
#include "ace.h"
#include "../ace_backend_api.h"

#ifdef OPENCL_AVAILABLE

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Internal structures
 * ============================================================================ */

typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char name[256];
    size_t total_mem;
    int compute_units;
    int max_threads;
} ocl_device_t;

typedef struct {
    cl_mem mem;
    size_t size;
} ocl_buffer_t;

typedef struct {
    cl_kernel kernel;
    cl_program program;
    char* name;
    ocl_device_t* device;
} ocl_kernel_t;

static cl_platform_id g_platform;

/* ============================================================================
 * ACE -> OpenCL translation
 * ============================================================================ */

static char* translate_to_opencl(const char* name, const char* src, const char* type_name) {
    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;
    
    char* p;
    while ((p = strstr(code, "T")) != NULL) {
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
    
    const char* params_start = strchr(code, '(');
    const char* params_end = strchr(code, ')');
    const char* body_start = strchr(code, '{');
    const char* body_end = strrchr(code, '}');

    if (!params_start || !params_end || !body_start || !body_end) {
        free(code);
        char* out = (char*)malloc(256);
        snprintf(out, 256, "__kernel void %s() { int GID = get_global_id(0); }\n", name);
        return out;
    }

    /* Process parameters - add __global to pointer types */
    size_t params_len = params_end - params_start + 1;
    char* new_params = (char*)malloc(params_len * 2);

    const char* p2 = params_start;
    char* dst = new_params;

    while (p2 <= params_end) {
        const char* star = strchr(p2, '*');
        const char* next_delim = strchr(p2, ',');
        const char* paren = strchr(p2, ')');

        if (!next_delim || (paren && paren < next_delim)) next_delim = paren;

        if (star && next_delim && star < next_delim) {
            /* Add __global before pointer parameter */
            const char* type_start = p2 + 1;
            while (type_start < star &&
                   (*type_start == ' ' || *type_start == '\t' ||
                    *type_start == ',' || *type_start == '(')) {
                *dst++ = *type_start++;
            }

            if (type_start < star) {
                strcpy(dst, "__global ");
                dst += 9;

                while (type_start <= params_end && *type_start != ',') {
                    *dst++ = *type_start++;
                    if (*type_start == ')') break;
                }
                p2 = type_start;
                continue;
            }
        }
        *dst++ = *p2++;
    }
    *dst = '\0';

    size_t body_len = body_end - body_start - 1;

    size_t total_len = strlen(name) + strlen(new_params) + body_len + 256;
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "__kernel void %s%s\n"
        "{\n"
        "    int GID = get_global_id(0);\n"
        "    %.*s\n"
        "}\n",
        name, new_params,
        (int)body_len, body_start + 1
    );

    free(new_params);
    free(code);
    return out;
}

/* ============================================================================
 * Backend operations
 * ============================================================================ */

static ace_error_t ocl_init(ace_backend_info_t* info) {
    cl_uint num_platforms;
    if (clGetPlatformIDs(1, &g_platform, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        printf("[OpenCL] No platforms found\n");
        return ACE_ERROR_BACKEND;
    }
    
    char name[128];
    clGetPlatformInfo(g_platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
    printf("[OpenCL] Backend initialized (platform: %s)\n", name);
    return ACE_OK;
}

static void ocl_shutdown(ace_backend_info_t* info) {
    /* OpenCL cleanup is automatic */
}

static ace_error_t ocl_device_count(int* count) {
    cl_uint num = 0;
    cl_int err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
    if (err != CL_SUCCESS || num == 0) {
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num);
        if (err != CL_SUCCESS) num = 0;
    }
    *count = num;
    return ACE_OK;
}

static ace_error_t ocl_device_get(int idx, void** dev) {
    cl_device_id devices[16];
    cl_uint num;
    
    cl_int err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 16, devices, &num);
    if (err != CL_SUCCESS || idx >= (int)num) {
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_CPU, 16, devices, &num);
        if (err != CL_SUCCESS || idx >= (int)num) {
            return ACE_ERROR_DEVICE;
        }
    }
    
    ocl_device_t* d = (ocl_device_t*)calloc(1, sizeof(*d));
    d->device = devices[idx];
    
    clGetDeviceInfo(d->device, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);
    
    cl_ulong mem;
    clGetDeviceInfo(d->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
    d->total_mem = mem;
    
    cl_uint cu;
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    d->compute_units = cu;
    
    size_t wg;
    clGetDeviceInfo(d->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
    d->max_threads = (int)wg;
    
    d->context = clCreateContext(NULL, 1, &d->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(d);
        return ACE_ERROR_DEVICE;
    }
    
    d->queue = clCreateCommandQueue(d->context, d->device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(d->context);
        free(d);
        return ACE_ERROR_DEVICE;
    }
    
    *dev = d;
    return ACE_OK;
}

static void ocl_device_release(void* dev) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (d) {
        if (d->queue) clReleaseCommandQueue(d->queue);
        if (d->context) clReleaseContext(d->context);
        free(d);
    }
}

static ace_error_t ocl_device_props(void* dev, void* props) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ace_device_props_t* p = (ace_device_props_t*)props;
    if (!d || !p) return ACE_ERROR_DEVICE;
    
    p->type = ACE_BACKEND_DEVICE_OPENCL;
    strncpy(p->name, d->name, sizeof(p->name) - 1);
    strcpy(p->vendor, "OpenCL");
    p->total_memory = d->total_mem;
    p->max_threads = d->max_threads;
    p->compute_units = d->compute_units;
    return ACE_OK;
}

static ace_error_t ocl_mem_alloc(void* dev, size_t size, void** ptr) {
    ocl_device_t* d = (ocl_device_t*)dev;
    
    cl_int err;
    cl_mem mem = clCreateBuffer(d->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) return ACE_ERROR_MEM;
    
    ocl_buffer_t* buf = (ocl_buffer_t*)calloc(1, sizeof(*buf));
    buf->mem = mem;
    buf->size = size;
    *ptr = buf;
    return ACE_OK;
}

static void ocl_mem_free(void* dev, void* ptr) {
    ocl_buffer_t* buf = (ocl_buffer_t*)ptr;
    if (buf) {
        clReleaseMemObject(buf->mem);
        free(buf);
    }
}

static ace_error_t ocl_mem_write(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)dst;
    cl_int err = clEnqueueWriteBuffer(d->queue, buf->mem, CL_TRUE, 0, size, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t ocl_mem_read(void* dev, void* dst, const void* src, size_t size) {
    ocl_device_t* d = (ocl_device_t*)dev;
    ocl_buffer_t* buf = (ocl_buffer_t*)src;
    cl_int err = clEnqueueReadBuffer(d->queue, buf->mem, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_IO;
}

static ace_error_t ocl_finish(void* dev) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_DEVICE;
    return (clFinish(d->queue) == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

static ace_error_t ocl_kernel_compile(void* dev, const char* name, const char* src,
                                       void** kernel, char** err_msg) {
    ocl_device_t* d = (ocl_device_t*)dev;

    /* 确定数据类型 */
    const char* type_name = "float";
    const char* suffix = strrchr(name, '_');
    if (suffix) {
        suffix++;
        if (strcmp(suffix, "int") == 0 || strcmp(suffix, "int32") == 0) type_name = "int";
        else if (strcmp(suffix, "double") == 0 || strcmp(suffix, "float64") == 0) type_name = "double";
        else if (strcmp(suffix, "long") == 0 || strcmp(suffix, "int64") == 0) type_name = "long";
        else if (strcmp(suffix, "half") == 0 || strcmp(suffix, "float16") == 0) type_name = "half";
        else if (strcmp(suffix, "char") == 0 || strcmp(suffix, "int8") == 0) type_name = "char";
        else if (strcmp(suffix, "uchar") == 0 || strcmp(suffix, "uint8") == 0) type_name = "uchar";
        else if (strcmp(suffix, "short") == 0 || strcmp(suffix, "int16") == 0) type_name = "short";
    }

    char* translated = translate_to_opencl(name, src, type_name);
    
    cl_int err;
    const char* srcs[1] = { translated };
    size_t lens[1] = { strlen(translated) };
    
    cl_program prog = clCreateProgramWithSource(d->context, 1, srcs, lens, &err);
    if (err != CL_SUCCESS) {
        free(translated);
        return ACE_ERROR_COMPILE;
    }
    
    err = clBuildProgram(prog, 1, &d->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, d->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (err_msg) {
            *err_msg = (char*)malloc(log_size + 1);
            clGetProgramBuildInfo(prog, d->device, CL_PROGRAM_BUILD_LOG, log_size, *err_msg, NULL);
        }
        clReleaseProgram(prog);
        free(translated);
        return ACE_ERROR_COMPILE;
    }
    
    cl_kernel krn = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        free(translated);
        if (err_msg) *err_msg = strdup("Failed to create kernel");
        return ACE_ERROR_COMPILE;
    }
    
    free(translated);
    
    ocl_kernel_t* k = (ocl_kernel_t*)calloc(1, sizeof(*k));
    k->kernel = krn;
    k->program = prog;
    k->name = strdup(name);
    k->device = d;
    
    *kernel = k;
    return ACE_OK;
}

static void ocl_kernel_release(void* kernel) {
    ocl_kernel_t* k = (ocl_kernel_t*)kernel;
    if (k) {
        if (k->kernel) clReleaseKernel(k->kernel);
        if (k->program) clReleaseProgram(k->program);
        free(k->name);
        free(k);
    }
}

static ace_error_t ocl_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                      ace_launch_config_t* cfg, void** args, size_t* sizes, int n) {
    ocl_device_t* d = (ocl_device_t*)dev;
    if (!d || !d->queue) return ACE_ERROR_LAUNCH;

    /* 查找缓存的内核（简化：每次重新编译，实际应该缓存） */
    const char* type_name = "float";
    const char* suffix = strrchr(kernel_def->name, '_');
    if (suffix) {
        suffix++;
        if (strcmp(suffix, "int") == 0 || strcmp(suffix, "int32") == 0) type_name = "int";
        else if (strcmp(suffix, "double") == 0 || strcmp(suffix, "float64") == 0) type_name = "double";
    }

    char* translated = translate_to_opencl(kernel_def->name, kernel_def->src, type_name);

    cl_int err;
    const char* srcs[1] = { translated };
    size_t lens[1] = { strlen(translated) };

    cl_program prog = clCreateProgramWithSource(d->context, 1, srcs, lens, &err);
    if (err != CL_SUCCESS) {
        free(translated);
        return ACE_ERROR_COMPILE;
    }

    err = clBuildProgram(prog, 1, &d->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        free(translated);
        return ACE_ERROR_COMPILE;
    }

    cl_kernel krn = clCreateKernel(prog, kernel_def->name, &err);
    clReleaseProgram(prog);
    free(translated);
    
    if (err != CL_SUCCESS) {
        return ACE_ERROR_COMPILE;
    }

    /* 设置参数 */
    for (int i = 0; i < n; i++) {
        if (sizes[i] == ACE_ARG_BUFFER) {
            ocl_buffer_t* buf = (ocl_buffer_t*)args[i];
            if (!buf || !buf->mem) {
                clReleaseKernel(krn);
                return ACE_ERROR_LAUNCH;
            }
            err = clSetKernelArg(krn, i, sizeof(cl_mem), &buf->mem);
        } else {
            err = clSetKernelArg(krn, i, sizeof(int), args[i]);
        }
        if (err != CL_SUCCESS) {
            clReleaseKernel(krn);
            return ACE_ERROR_LAUNCH;
        }
    }

    size_t global[3] = { cfg->grid[0] * cfg->block[0], cfg->grid[1] * cfg->block[1], cfg->grid[2] * cfg->block[2] };
    size_t local[3] = { cfg->block[0], cfg->block[1], cfg->block[2] };

    err = clEnqueueNDRangeKernel(d->queue, krn, 3, NULL, global, local, 0, NULL, NULL);
    clReleaseKernel(krn);
    
    return (err == CL_SUCCESS) ? ACE_OK : ACE_ERROR_LAUNCH;
}

/* Backend registration */
static ace_backend_ops_t ocl_ops = {
    .init = ocl_init,
    .shutdown = ocl_shutdown,
    .device_count = ocl_device_count,
    .device_get = ocl_device_get,
    .device_release = ocl_device_release,
    .device_props = ocl_device_props,
    .mem_alloc = ocl_mem_alloc,
    .mem_free = ocl_mem_free,
    .mem_write = ocl_mem_write,
    .mem_read = ocl_mem_read,
    .finish = ocl_finish,
    .kernel_launch = ocl_kernel_launch,
};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_OPENCL, "OpenCL", &ocl_ops)

#else

/* OpenCL SDK not available - provide stub */
static ace_backend_ops_t ocl_ops = {0};

ACE_DEFINE_BACKEND(ACE_BACKEND_DEVICE_OPENCL, "OpenCL (unavailable)", &ocl_ops)

#endif /* OPENCL_AVAILABLE */