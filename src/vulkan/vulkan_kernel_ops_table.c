/**
 * @file vulkan_kernel_ops_table.c
 * @brief Vulkan 内核操作表实现
 *
 * 设计原则：高内聚，低耦合
 * - 每个操作的完整实现都在一个函数内
 * - 函数内部根据 type->needs_emulation 和 type->fn_to_f32 决定生成什么代码
 */
#include "vulkan_kernel_ops_table.h"
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * 算术运算注入函数
 * ============================================================================ */

static int inject_add(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kadd(a, b) ((a) + (b))\n");
    }
    /* 需要模拟 */
    if (type->fn_to_f32 && type->fn_from_f32) {
        /* FP16/BF16：使用转换函数 */
        return sprintf(buf, "%s kadd(%s a, %s b) { return %s(%s(a) + %s(b)); }\n",
                       type->name, type->name, type->name,
                       type->fn_from_f32, type->fn_to_f32, type->fn_to_f32);
    }
    /* 通用模拟 */
    return sprintf(buf, "%s kadd(%s a, %s b) { return (a) + (b); }\n",
                   type->name, type->name, type->name);
}

static int inject_sub(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define ksub(a, b) ((a) - (b))\n");
    }
    if (type->fn_to_f32 && type->fn_from_f32) {
        return sprintf(buf, "%s ksub(%s a, %s b) { return %s(%s(a) - %s(b)); }\n",
                       type->name, type->name, type->name,
                       type->fn_from_f32, type->fn_to_f32, type->fn_to_f32);
    }
    return sprintf(buf, "%s ksub(%s a, %s b) { return (a) - (b); }\n",
                   type->name, type->name, type->name);
}

static int inject_mul(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kmul(a, b) ((a) * (b))\n");
    }
    if (type->fn_to_f32 && type->fn_from_f32) {
        return sprintf(buf, "%s kmul(%s a, %s b) { return %s(%s(a) * %s(b)); }\n",
                       type->name, type->name, type->name,
                       type->fn_from_f32, type->fn_to_f32, type->fn_to_f32);
    }
    return sprintf(buf, "%s kmul(%s a, %s b) { return (a) * (b); }\n",
                   type->name, type->name, type->name);
}

static int inject_div(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kdiv(a, b) ((a) / (b))\n");
    }
    if (type->fn_to_f32 && type->fn_from_f32) {
        return sprintf(buf, "%s kdiv(%s a, %s b) { return %s(%s(a) / %s(b)); }\n",
                       type->name, type->name, type->name,
                       type->fn_from_f32, type->fn_to_f32, type->fn_to_f32);
    }
    return sprintf(buf, "%s kdiv(%s a, %s b) { return (a) / (b); }\n",
                   type->name, type->name, type->name);
}

/* ============================================================================
 * 比较运算注入函数
 * ============================================================================ */

static int inject_lt(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define klt(a, b) ((a) < (b))\n");
    }
    return sprintf(buf, "bool klt(%s a, %s b) { return (a) < (b); }\n",
                   type->name, type->name);
}

static int inject_le(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kle(a, b) ((a) <= (b))\n");
    }
    return sprintf(buf, "bool kle(%s a, %s b) { return (a) <= (b); }\n",
                   type->name, type->name);
}

static int inject_gt(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kgt(a, b) ((a) > (b))\n");
    }
    return sprintf(buf, "bool kgt(%s a, %s b) { return (a) > (b); }\n",
                   type->name, type->name);
}

static int inject_ge(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kge(a, b) ((a) >= (b))\n");
    }
    return sprintf(buf, "bool kge(%s a, %s b) { return (a) >= (b); }\n",
                   type->name, type->name);
}

static int inject_eq(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define keq(a, b) ((a) == (b))\n");
    }
    return sprintf(buf, "bool keq(%s a, %s b) { return (a) == (b); }\n",
                   type->name, type->name);
}

static int inject_ne(char* buf, const dtype_info_t* type) {
    if (!type->needs_emulation) {
        return sprintf(buf, "#define kne(a, b) ((a) != (b))\n");
    }
    return sprintf(buf, "bool kne(%s a, %s b) { return (a) != (b); }\n",
                   type->name, type->name);
}

/* ============================================================================
 * 内核操作表
 * ============================================================================ */

static kernel_op_t g_ops_table[] = {
    {"kadd", inject_add},
    {"ksub", inject_sub},
    {"kmul", inject_mul},
    {"kdiv", inject_div},
    {"klt",  inject_lt},
    {"kle",  inject_le},
    {"kgt",  inject_gt},
    {"kge",  inject_ge},
    {"keq",  inject_eq},
    {"kne",  inject_ne},
    {NULL,   NULL}
};

const kernel_op_t* get_kernel_ops_table(void) {
    return g_ops_table;
}

const kernel_op_t* find_kernel_op(const char* name) {
    if (!name) return NULL;
    for (int i = 0; g_ops_table[i].name; i++) {
        if (strcmp(g_ops_table[i].name, name) == 0) {
            return &g_ops_table[i];
        }
    }
    return NULL;
}
