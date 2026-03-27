/**
 * @file vulkan_type_utils.c
 * @brief Vulkan 类型辅助工具 - 基于设备级别类型表
 *
 * 设计说明:
 * - 所有类型相关信息在 dtype_info_t 表中
 * - 所有内核操作在 kernel_op_t 表中
 * - 本文件只负责：扫描代码、查表、调用注入函数
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vulkan/vulkan.h>
#include "ace.h"
#include "vulkan_backend.h"
#include "vulkan_kernel_ops_table.h"

/* ============================================================================
 * 公共 API：GLSL 代码翻译
 * ============================================================================ */

char* vk_translate_to_glsl(const vk_device_internal_t* dev, const char* name, const char* src, 
                           ace_dtype_t dtype, int* n_buffers, int* n_scalars) {
    (void)name;
    const dtype_info_t* type = vk_dtype_info(&dev->dtype_table, dtype);
    
    /* 解析函数体 */
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) {
        *n_buffers = 0;
        *n_scalars = 0;
        return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    }

    /* 解析参数 */
    typedef struct { char name[64]; int is_buffer; } param_info_t;
    param_info_t params[16];
    int n_params = 0;
    *n_buffers = 0;
    *n_scalars = 0;

    const char* p = strchr(src, '(');
    if (p) {
        p++;
        while (*p && *p != ')') {
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
            if (*p == ')' || *p == '{') break;

            param_info_t* param = &params[n_params];
            char* dst = param->name;
            int is_ptr = 0;
            int in_type = 1;

            while (*p && *p != ',' && *p != ')' && n_params < 16) {
                if (*p == '*') { is_ptr = 1; in_type = 0; p++; continue; }
                if (in_type) { if (*p == ' ' || *p == '\t' || *p == '\n') in_type = 0; p++; continue; }
                if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '&') *dst++ = *p;
                p++;
            }
            *dst = '\0';

            param->is_buffer = is_ptr;
            if (is_ptr) (*n_buffers)++; else (*n_scalars)++;
            n_params++;
            if (*p == ',') p++;
        }
    }

    /* 构建 push constants */
    char push_constants[1024] = "";
    char pc_access[1024] = "";
    if (*n_scalars > 0) {
        strcpy(push_constants, "layout(push_constant) uniform PC {\n");
        int scalar_idx = 0;
        for (int i = 0; i < n_params; i++) {
            if (!params[i].is_buffer) {
                char line[128];
                const char* scalar_type = (scalar_idx == 0) ? "int" : type->name;
                snprintf(line, sizeof(line), "  %s s%d;\n", scalar_type, scalar_idx);
                strcat(push_constants, line);
                char access[128];
                snprintf(access, sizeof(access), "#define %s pc.s%d\n", params[i].name, scalar_idx);
                strcat(pc_access, access);
                scalar_idx++;
            }
        }
        strcat(push_constants, "} pc;\n");
    }

    /* 构建 buffers */
    char buffers[2048] = "";
    int buf_idx = 0;
    for (int i = 0; i < n_params && buf_idx < *n_buffers; i++) {
        if (params[i].is_buffer) {
            char buf_decl[256];
            snprintf(buf_decl, sizeof(buf_decl),
                "layout(binding = %d, std430) buffer B%d { %s d%d[]; };\n",
                buf_idx, buf_idx, type->name, buf_idx);
            strcat(buffers, buf_decl);
            char def[128];
            snprintf(def, sizeof(def), "#define %s d%d\n", params[i].name, buf_idx);
            strcat(buffers, def);
            buf_idx++;
        }
    }

    /* 提取函数体 */
    size_t body_len = body_end - body_start - 1;
    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';

    /* 构建基础 GLSL 代码 */
    size_t len = 4096 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) +
                 strlen(type->extensions) + strlen(type->type_def) + strlen(body);
    char* out = (char*)malloc(len);

    snprintf(out, len,
        "#version 450\n"
        "%s"
        "%s"
        "layout(local_size_x = 256) in;\n"
        "#define T %s\n"
        "%s"
        "%s"
        "%s"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "%s\n%s\n%s\n"
        "void main() {\n"
        "%s"
        "}\n",
        type->extensions,
        type->type_def,
        type->name,
        buffers,
        push_constants,
        pc_access,
        type->k_zero,
        type->k_one,
        type->k_neg_one,
        body
    );

    free(body);

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
                        const kernel_op_t* op = find_kernel_op(func_name);
                        if (op && op->inject) {
                            /* 查找注入位置：在类型定义之后，void main() 之前 */
                            char* insert_pos = strstr(out, "void main()");
                            if (!insert_pos) insert_pos = out + strlen(out);
                            
                            /* 注入操作实现 */
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