/**
 * @file opencl_type_utils.c
 * @brief OpenCL 类型辅助工具 - 基于表驱动架构
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <CL/cl.h>
#include "ace.h"
#include "opencl_backend.h"
#include "opencl_dtype_table.h"
#include "opencl_kernel_ops_table.h"

char* ocl_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const dtype_info_t* type = opencl_dtype_info(dtype);
    
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');

    if (!body_start || !body_end) {
        char* out = (char*)malloc(256);
        snprintf(out, 256, "%s__kernel void %s() { int GID = get_global_id(0); }\n",
                 type->extension, name);
        return out;
    }

    size_t body_len = body_end - body_start - 1;

    /* 构建基础代码框架 */
    size_t total_len = strlen(name) + body_len + 4096 +
                       strlen(type->extension) + strlen(type->k_zero);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "%s\n%s\n%s\n"
        "#define T %s\n"
        "#define GID get_global_id(0)\n"
        "#define LID get_local_id(0)\n"
        "#define BSIZE get_local_size(0)\n",
        type->extension,
        type->k_zero,
        type->k_one,
        type->k_neg_one,
        type->name
    );

    /* 处理参数列表，添加 __global 限定符 */
    const char* params_start = strchr(src, '(');
    const char* params_end = strchr(src, ')');
    if (params_start && params_end) {
        strcat(out, "__kernel void ");
        strcat(out, name);
        
        /* 简单处理：直接复制参数列表，在 * 前添加 __global */
        const char* p = params_start;
        while (p <= params_end) {
            if (*p == '*' && p > params_start && *(p-1) != '*' && *(p+1) != '*') {
                strcat(out, " __global");
            }
            char ch[2] = {*p, '\0'};
            strcat(out, ch);
            p++;
        }
        strcat(out, "\n");
    }

    /* 追加函数体 */
    strcat(out, body_start);
    strcat(out, "\n");

    /* 非原生支持时注入类型定义 */
    if (type->needs_emulation && type->type_def && type->type_def[0]) {
        char* insert_pos = strstr(out, type->extension);
        if (insert_pos) {
            insert_pos += strlen(type->extension);
            size_t prefix_len = insert_pos - out;
            size_t new_len = strlen(out) + strlen(type->type_def) + 10;
            char* temp = (char*)malloc(new_len);
            if (temp) {
                strncpy(temp, out, prefix_len);
                temp[prefix_len] = '\0';
                strcat(temp, type->type_def);
                strcat(temp, "\n");
                strcat(temp, insert_pos);
                free(out);
                out = temp;
            }
        }
    }

    /* 扫描并注入内核操作 */
    const char* s = src;
    while (*s) {
        if (islower(*s) && s[1] && islower(s[1]) && s[2] && islower(s[2])) {
            const char* start = s;
            while (islower(*s)) s++;
            if (*s == '(') {
                char func_name[16];
                size_t len = s - start;
                if (len < sizeof(func_name)) {
                    strncpy(func_name, start, len);
                    func_name[len] = '\0';
                    if (func_name[0] == 'k' && func_name[1] && func_name[2]) {
                        const kernel_op_t* op = opencl_find_kernel_op(func_name);
                        if (op && op->inject) {
                            char* insert_pos = strstr(out, "__kernel");
                            if (!insert_pos) insert_pos = out + strlen(out);
                            
                            char inject_buf[512];
                            int inj_len = op->inject(inject_buf, type);
                            if (inj_len > 0) {
                                size_t prefix_len = insert_pos - out;
                                size_t new_len = strlen(out) + inj_len + 10;
                                char* temp = (char*)malloc(new_len);
                                if (temp) {
                                    strncpy(temp, out, prefix_len);
                                    temp[prefix_len] = '\0';
                                    strncat(temp, inject_buf, inj_len);
                                    strcat(temp, "\n");
                                    strcat(temp, insert_pos);
                                    free(out);
                                    out = temp;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            s++;
        }
    }

    return out;
}
