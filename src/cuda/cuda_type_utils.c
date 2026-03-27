/**
 * @file cuda_type_utils.c
 * @brief CUDA 类型辅助工具 - 基于表驱动架构
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ace.h"
#include "cuda_dtype_table.h"
#include "cuda_kernel_ops_table.h"

char* cuda_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const dtype_info_t* type = cuda_dtype_info(dtype);
    
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');

    if (!body_start || !body_end) {
        char* out = (char*)malloc(256);
        snprintf(out, 256, "extern \"C\" __global__ void %s() {}\n", name);
        return out;
    }

    size_t body_len = body_end - body_start - 1;

    /* 构建基础代码框架 */
    size_t total_len = strlen(name) + body_len + 4096 +
                       strlen(type->headers) + strlen(type->k_zero);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "%s\n%s\n%s\n"
        "#define T %s\n"
        "#define GID (blockIdx.x * blockDim.x + threadIdx.x)\n"
        "#define LID threadIdx.x\n"
        "#define BSIZE blockDim.x\n"
        "extern \"C\" __global__ void %s%s\n",
        type->headers,
        type->k_zero,
        type->k_one,
        type->k_neg_one,
        type->name,
        name, strchr(src, '(')
    );

    /* 非原生支持时注入类型定义 */
    if (type->needs_emulation && type->type_def && type->type_def[0]) {
        char* insert_pos = strstr(out, type->headers);
        if (insert_pos) {
            insert_pos += strlen(type->headers);
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
                        const kernel_op_t* op = cuda_find_kernel_op(func_name);
                        if (op && op->inject) {
                            char* insert_pos = strstr(out, "extern \"C\"");
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
