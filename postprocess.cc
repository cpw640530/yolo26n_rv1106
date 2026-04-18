/**
 * YOLO26 模型后处理 C++ 实现
 *
 * 该文件实现 YOLO26 目标检测模型在 Rockchip RV1103/RV1106 平台上的后处理逻辑：
 *   - 支持 INT8 量化模型和 FP32 模型
 *   - 支持 Distribution Focal Loss (DFL) 解码
 *   - 支持 Non-Maximum Suppression (NMS) 去重
 *   - 支持多种数据布局：NHWC、NCHW、CHW
 *
 * 版权: Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
 * 许可证: Apache License, Version 2.0
 */

#include "yolov8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

// COCO 80 类别的标签文件路径
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

// 存储类别标签的指针数组
static char *labels[OBJ_CLASS_NUM];

/**
 * clamp 函数：将值限制在 [min, max] 范围内
 * 相当于 std::clamp，但实现更简单
 */
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

/**
 * readLine: 从文件中读取一行文本
 *
 * 该函数动态分配内存来存储读取的行，支持不定长度的行
 * 函数内部会不断 realloc 直至遇到换行符或 EOF
 *
 * 参数:
 *   fp: 文件指针
 *   buffer: 输出参数，指向读取到的字符串buffer
 *   len: 输出参数，存储读取到的字符串长度
 *
 * 返回:
 *   成功: 指向 buffer 的指针
 *   失败: NULL
 */
static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    // 初始分配 1 字节（后续会 realloc 扩展）
    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // 内存分配失败

    // 逐字符读取，直到遇到换行符或文件结束
    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // 内存分配失败
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';  // 字符串结束符

    *len = buff_len;

    // 检测是否真正到达文件末尾
    // 情况1: 读到 EOF 但从未读取任何字符（空文件）
    // 情况2: 读取过程中发生文件错误（ferror）
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

/**
 * readLines: 从文件中读取多行文本
 *
 * 打开指定文件，逐行读取所有内容并存储到 lines 数组中
 *
 * 参数:
 *   fileName: 要读取的文件路径
 *   lines: 存储读取结果的字符指针数组（输出参数）
 *   max_line: lines 数组的最大容量
 *
 * 返回:
 *   实际读取的行数，失败返回 -1
 */
static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("打开文件失败: %s\n", fileName);
        return -1;
    }

    // 循环读取每一行，直到文件结束或达到最大行数
    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

/**
 * loadLabelName: 加载类别标签名称文件
 *
 * 从文件中读取 COCO 80 类的名称，存储到 label 数组中
 *
 * 参数:
 *   locationFilename: 标签文件路径
 *   label: 存储标签的字符指针数组（输出参数）
 *
 * 返回:
 *   0 = 成功，失败返回 -1
 */
static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("加载类别标签文件: %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

/**
 * CalculateOverlap: 计算两个矩形框的 IoU（Intersection over Union）
 *
 * IoU 是目标检测中衡量两个框重叠程度的指标
 * IoU = 重叠面积 / 合并面积 = 重叠面积 / (两个框面积之和 - 重叠面积)
 *
 * 参数:
 *   xmin0, ymin0, xmax0, ymax0: 第一个框的左上角和右下角坐标
 *   xmin1, ymin1, xmax1, ymax1: 第二个框的左上角和右下角坐标
 *
 * 返回:
 *   两个框的 IoU 值，范围 [0, 1]
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    // 计算重叠区域的宽度：
    // fmin(xmax0, xmax1) - fmax(xmin0, xmin1) 得到重叠区域宽度
    // +1 是因为像素坐标是闭区间，例如从 0 到 10 代表 11 个像素
    // fmax(0, ...) 确保重叠宽度不为负数（无重叠时）
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);

    // 计算重叠区域的高度
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);

    // 重叠面积
    float i = w * h;

    // 合并面积 = 框1面积 + 框2面积 - 重叠面积
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;

    // 避免除零错误：无重叠时返回 0
    return u <= 0.f ? 0.f : (i / u);
}

/**
 * nms: Non-Maximum Suppression 非极大值抑制
 *
 * 针对特定类别的检测结果执行 NMS，去除重叠的检测框
 * 算法流程：
 *   1. 遍历所有检测框（按置信度降序排列）
 *   2. 保留当前最高置信度的框
 *   3. 计算它与所有剩余框的 IoU
 *   4. 若 IoU > 阈值，将该框标记为删除（order[j] = -1）
 *   5. 重复直至所有框处理完毕
 *
 * 参数:
 *   validCount: 有效检测框的数量
 *   outputLocations: 检测框坐标数组，存储格式 [x1, y1, w, h, x1, y1, w, h, ...]
 *   classIds: 每个检测框对应的类别 ID 数组
 *   order: 按置信度降序排列的检测框索引数组
 *   filterId: 要执行 NMS 的目标类别 ID
 *   threshold: IoU 阈值，超过此值的框将被过滤
 *
 * 返回:
 *   0 = 成功
 */
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        // 跳过已删除的框（order[j] = -1）或不属于当前类别的框
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        // 与后续所有框进行比较
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }

            // 提取框 n 的坐标 [x1, y1, w, h]
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];  // x1 + w
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3]; // y1 + h

            // 提取框 m 的坐标 [x1, y1, w, h]
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            // 计算两个框的 IoU
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            // 若 IoU 超过阈值，将框 m 标记为删除
            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

/**
 * quick_sort_indice_inverse: 快速排序（降序）
 *
 * 基于输入数组的值进行降序排序，同时调整索引数组以跟踪原始位置
 * 这是一种"间接排序"：不直接排序 input 数组，而是通过 indices
 * 数组记录排序后的顺序
 *
 * 参数:
 *   input: 要排序的浮点数组（这里存储的是置信度分数）
 *   left: 排序区间的左边界
 *   right: 排序区间的右边界
 *   indices: 索引数组，初始为 [0, 1, 2, ..., n-1]，排序后保持对应关系
 *
 * 返回:
 *   基准元素最终所在的位置
 */
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            // 从右向左找到第一个小于基准的值
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            // 从左向右找到第一个大于基准的值
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        // 将基准放到最终位置
        input[low] = key;
        indices[low] = key_index;
        // 递归排序左右两部分
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

/**
 * sigmoid: Sigmoid 激活函数
 * 将任意实数映射到 (0, 1) 区间，常用于将分数转换为概率
 */
static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

/**
 * unsigmoid: Sigmoid 的反函数
 * 将 (0, 1) 区间的概率值还原为实数，用于置信度阈值转换
 */
static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

/**
 * __clip: 将浮点值限制在指定范围内
 * 用于量化/反量化时确保值不超出目标数据类型范围
 */
inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

/**
 * qnt_f32_to_affine: 将 FP32 值量化为 INT8 值（带零点偏移）
 *
 * 量化公式：qnt = round(f32 / scale + zp)
 * 这是典型的 affine quantization（仿射量化）方式
 *
 * 参数:
 *   f32: 待量化的 FP32 浮点值
 *   zp: 零点（zero point），量化偏移量
 *   scale: 量化尺度
 *
 * 返回:
 *   量化后的 INT8 值，范围 [-128, 127]
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

/**
 * qnt_f32_to_affine_u8: 将 FP32 值量化为 UINT8 值（带零点偏移）
 * 用于需要无符号量化的情况（如某些平台的分类分数）
 */
static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

/**
 * deqnt_affine_to_f32: 将 INT8 量化值反量化为 FP32 值
 *
 * 反量化公式：f32 = (qnt - zp) * scale
 *
 * 参数:
 *   qnt: INT8 量化值
 *   zp: 零点（zero point）
 *   scale: 量化尺度
 *
 * 返回:
 *   反量化后的 FP32 浮点值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

/**
 * deqnt_affine_u8_to_f32: 将 UINT8 量化值反量化为 FP32 值
 */
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

/* ===================== YOLO26 专用后处理（纯 C 实现） ===================== */

/**
 * DetectRectYolo26: YOLO26 检测结果结构体
 *
 * 存储单个检测框的完整信息：坐标、置信度和类别
 */
typedef struct
{
    float xmin;     // 检测框左上角 x 坐标
    float ymin;     // 检测框左上角 y 坐标
    float xmax;     // 检测框右下角 x 坐标
    float ymax;     // 检测框右下角 y 坐标
    float score;    // 置信度得分
    int classId;    // 类别 ID
} DetectRectYolo26;

/**
 * fast_exp: 快速指数函数（近似计算）
 *
 * 使用位操作技巧近似计算 e^x，比标准 expf() 快数倍
 * 通过将 x 映射到 IEEE 754 浮点数的尾数域实现
 *
 * 原理：e^x 可以写成 2^(x / ln(2))
 * 通过线性变换将 x 转换为浮点数的位表示
 *
 * 参数:
 *   x: 指数输入值
 *
 * 返回:
 *   e^x 的近似值
 */
static inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    // 系数由实验得出，用于最小化近似误差
    v.i = (uint32_t)(12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

/**
 * DeQnt2F32: INT8 量化值转 FP32 浮点数
 *
 * 量化公式：f32 = (qnt - zp) * scale
 * 这是 RKNN 模型输出反量化的标准方式
 */
static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/**
 * yolo26_sigmoid: YOLO26 专用的 Sigmoid 激活函数
 *
 * 使用快速指数函数替代标准 expf()，提升计算效率
 * Sigmoid 函数：σ(x) = 1 / (1 + e^(-x))
 */
static float yolo26_sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

/* 动态 yolo26 解码（不依赖固定输出顺序/固定 mapSize）：
 * reg: [4, H, W] 或 [H, W, 4] 这种按 yolo26 demo 的排布，这里按 (c * H * W + h * W + w) 访问
 * cls: [C, H, W]，同样按 (c * H * W + h * W + w) 访问
 */
/**
 * yolo26_decode_head_chw: YOLO26 检测头解码（CHW 数据布局）
 *
 * CHW 布局：通道优先存储，即所有通道的同一位置像素连续存放
 * 地址计算：cls[c*H*W + h*W + w], reg[k*H*W + h*W + w]
 *
 * 参数:
 *   reg: 回归分支输出数据（INT8 量化格式）
 *   reg_zp: 回归分支的零点偏移
 *   reg_scale: 回归分支的量化尺度
 *   cls: 分类分支输出数据（INT8 量化格式）
 *   cls_zp: 分类分支的零点偏移
 *   cls_scale: 分类分支的量化尺度
 *   grid_h: 特征图高度
 *   grid_w: 特征图宽度
 *   stride: 当前检测头的步长（8/16/32）
 *   threshold: 置信度阈值
 *   rects: 输出参数，存储检测结果的数组
 *   max_rects: rects 数组的最大容量
 *   rect_count: 输出参数，已检测到的框数量
 *
 * 返回:
 *   0 = 成功
 */
static int yolo26_decode_head_chw(int8_t *reg, int reg_zp, float reg_scale,
                                  int8_t *cls, int cls_zp, float cls_scale,
                                  int grid_h, int grid_w, int stride,
                                  float threshold,
                                  DetectRectYolo26 *rects, int max_rects, int *rect_count)
{
    int hw = grid_h * grid_w;  // 特征图上总像素点数
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            float cls_max = -1e9f;  // 初始化为很小的值
            int cls_index = -1;     // 记录最高分数的类别 ID

            int offset_hw = h * grid_w + w;  // 当前像素在一维数组中的偏移

            // 遍历所有类别，找到分数最高的那个
            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                // CHW 布局：第 c 个通道的数据起始位置是 c*hw
                float v = (float)cls[c * hw + offset_hw];
                if (c == 0 || v > cls_max)
                {
                    cls_max = v;
                    cls_index = c;
                }
            }

            // 反量化并计算置信度分数
            float score = yolo26_sigmoid(DeQnt2F32((int8_t)cls_max, cls_zp, cls_scale));
            if (score <= threshold)
            {
                continue;  // 低于阈值，跳过此检测点
            }

            // 反量化回归值：tx, ty, tw, th
            float cx = DeQnt2F32(reg[0 * hw + offset_hw], reg_zp, reg_scale);
            float cy = DeQnt2F32(reg[1 * hw + offset_hw], reg_zp, reg_scale);
            float cw = DeQnt2F32(reg[2 * hw + offset_hw], reg_zp, reg_scale);
            float ch = DeQnt2F32(reg[3 * hw + offset_hw], reg_zp, reg_scale);

            // 计算特征图上对应像素的坐标（中心点）
            float center_x = (float)w + 0.5f;
            float center_y = (float)h + 0.5f;

            // YOLO 解码公式：将回归值转换为绝对坐标
            // xmin = (grid_x - tx) * stride
            // ymin = (grid_y - ty) * stride
            // xmax = (grid_x + tw) * stride
            // ymax = (grid_y + th) * stride
            float xmin = (center_x - cx) * stride;
            float ymin = (center_y - cy) * stride;
            float xmax = (center_x + cw) * stride;
            float ymax = (center_y + ch) * stride;

            // 检查是否超出最大检测框数量限制
            if (*rect_count >= max_rects)
            {
                continue;
            }

            // 存储检测结果
            rects[*rect_count].xmin = xmin;
            rects[*rect_count].ymin = ymin;
            rects[*rect_count].xmax = xmax;
            rects[*rect_count].ymax = ymax;
            rects[*rect_count].classId = cls_index;
            rects[*rect_count].score = score;
            (*rect_count)++;
        }
    }

    return 0;
}

/* yolo26 head 解码（按 NHWC 排布访问）：cls[hw*C + c], reg[hw*4 + k]
 * RV1106/1103 的输出通常是 NHWC（参考原 yolov8 process_i8_rv1106 的访问方式）
 */
/**
 * yolo26_decode_head_nhwc: YOLO26 检测头解码（NHWC 数据布局）
 *
 * NHWC 布局：每个像素点的所有通道连续存放
 * 地址计算：cls[h*W*C + w*C + c], reg[h*W*4 + w*4 + k]
 *
 * RV1106/1103 平台的 NPU 输出通常采用 NHWC 格式
 *
 * 参数:
 *   reg: 回归分支输出数据（INT8 量化格式）
 *   reg_zp: 回归分支的零点偏移
 *   reg_scale: 回归分支的量化尺度
 *   cls: 分类分支输出数据（INT8 量化格式）
 *   cls_zp: 分类分支的零点偏移
 *   cls_scale: 分类分支的量化尺度
 *   grid_h: 特征图高度
 *   grid_w: 特征图宽度
 *   stride: 当前检测头的步长（8/16/32）
 *   threshold: 置信度阈值
 *   rects: 输出参数，存储检测结果的数组
 *   max_rects: rects 数组的最大容量
 *   rect_count: 输出参数，已检测到的框数量
 *
 * 返回:
 *   0 = 成功
 */
static int yolo26_decode_head_nhwc(int8_t *reg, int reg_zp, float reg_scale,
                                   int8_t *cls, int cls_zp, float cls_scale,
                                   int grid_h, int grid_w, int stride,
                                   float threshold,
                                   DetectRectYolo26 *rects, int max_rects, int *rect_count)
{
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            int offset_hw = h * grid_w + w;  // 当前像素在一行中的偏移

            float cls_max = -1e9f;
            int cls_index = -1;
            // NHWC 布局：每个像素的 C 个类别分数连续存放
            int cls_base = offset_hw * OBJ_CLASS_NUM;
            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                float v = (float)cls[cls_base + c];
                if (c == 0 || v > cls_max)
                {
                    cls_max = v;
                    cls_index = c;
                }
            }

            // 反量化并计算置信度分数
            float score = yolo26_sigmoid(DeQnt2F32((int8_t)cls_max, cls_zp, cls_scale));
            if (score <= threshold)
            {
                continue;
            }

            // NHWC 布局：每个像素的 4 个回归值连续存放
            int reg_base = offset_hw * 4;
            float cx = DeQnt2F32(reg[reg_base + 0], reg_zp, reg_scale);
            float cy = DeQnt2F32(reg[reg_base + 1], reg_zp, reg_scale);
            float cw = DeQnt2F32(reg[reg_base + 2], reg_zp, reg_scale);
            float ch = DeQnt2F32(reg[reg_base + 3], reg_zp, reg_scale);

            // 计算特征图上对应像素的坐标（中心点）
            float center_x = (float)w + 0.5f;
            float center_y = (float)h + 0.5f;

            // YOLO 解码公式
            float xmin = (center_x - cx) * stride;
            float ymin = (center_y - cy) * stride;
            float xmax = (center_x + cw) * stride;
            float ymax = (center_y + ch) * stride;

            if (*rect_count >= max_rects)
            {
                continue;
            }

            // 存储检测结果
            rects[*rect_count].xmin = xmin;
            rects[*rect_count].ymin = ymin;
            rects[*rect_count].xmax = xmax;
            rects[*rect_count].ymax = ymax;
            rects[*rect_count].classId = cls_index;
            rects[*rect_count].score = score;
            (*rect_count)++;
        }
    }

    return 0;
}

/* ===================== YOLO26 专用后处理结束 ===================== */

/**
 * compute_dfl: Distribution Focal Loss (DFL) 计算
 *
 * DFL 是 YOLO 系列模型中用于边界框回归的损失函数
 * 它将离散的回归值通过 softmax 加权求和转换为连续值
 *
 * 原理：给定 4 个子区间（dfl_len=4）的 logits 值
 * 使用 softmax 得到概率分布，然后计算期望值作为最终回归结果
 *
 * 参数:
 *   tensor: 输入张量，存储格式 [logit0_0~3, logit1_0~3, logit2_0~3, logit3_0~3]
 *           即 4 个坐标通道，每个通道 4 个区间
 *   dfl_len: 每个坐标的分布区间数量（通常为 4）
 *   box: 输出数组，存储解码后的 4 个回归值 (x, y, w, h)
 */
static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];   // 存储每个区间的 exp 值
        float exp_sum=0;        // exp 值总和
        float acc_sum=0;        // 加权累积和

        // 第一步：计算各区间的 exp 值并求和
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }

        // 第二步：计算加权累积和（即期望值）
        // 期望 = Σ(概率_i × 区间索引_i)
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;  // 存储第 b 个坐标的 DFL 解码结果
    }
}

/**
 * process_u8: 处理 UINT8 量化格式的 YOLO 模型输出
 *
 * 适用于 RKNPU1 等支持 UINT8 量化的平台
 * 处理流程：
 *   1. 遍历特征图上的每个像素点
 *   2. 找出置信度最高的类别
 *   3. 低于阈值的点直接跳过
 *   4. 对高分点执行 DFL 解码得到边界框
 *   5. 将边界框坐标和置信度存入输出向量
 *
 * 参数:
 *   box_tensor: 边界框回归张量（UINT8 量化）
 *   box_zp: 边界框张量的零点偏移
 *   box_scale: 边界框张量的量化尺度
 *   score_tensor: 类别置信度张量（UINT8 量化）
 *   score_zp: 置信度张量的零点偏移
 *   score_scale: 置信度张量的量化尺度
 *   score_sum_tensor: 置信度求和张量（可选，用于快速过滤）
 *   score_sum_zp: 求和张量的零点偏移
 *   score_sum_scale: 求和张量的量化尺度
 *   grid_h, grid_w: 特征图尺寸
 *   stride: 检测头步长
 *   dfl_len: DFL 分布区间长度
 *   boxes: 输出参数，存储边界框坐标 [x1, y1, w, h, ...]
 *   objProbs: 输出参数，存储置信度分数
 *   classId: 输出参数，存储类别 ID
 *   threshold: 置信度阈值
 *
 * 返回:
 *   validCount: 有效检测框数量
 */
static int process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                      uint8_t *score_tensor, int32_t score_zp, float score_scale,
                      uint8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // 将 FP32 阈值转换为 UINT8 量化值，用于快速比较
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8(threshold, score_zp, score_scale);
    uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 使用 score_sum 快速过滤：若总分数低于阈值，直接跳过该点
            // 这种预过滤可以减少后续计算量
            if (score_sum_tensor != nullptr)
            {
                if (score_sum_tensor[offset] < score_sum_thres_u8)
                {
                    continue;
                }
            }

            // 遍历所有类别，找到最高分的类别
            uint8_t max_score = -score_zp;  // 初始化为最小可能值
            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;  // 移动到下一通道的同一位置
            }

            // 如果最高分超过阈值，执行边界框解码
            if (max_score > score_thres_u8)
            {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];

                // 读取 DFL 数据并反量化
                for (int k = 0; k < dfl_len * 4; k++)
                {
                    before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }

                // 执行 DFL 解码
                compute_dfl(before_dfl, dfl_len, box);

                // 计算最终边界框坐标
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;

                // 存储结果
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                // 反量化并存储置信度和类别
                objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

/**
 * process_i8: 处理 INT8 量化格式的 YOLO 模型输出
 *
 * 与 process_u8 类似，但使用 INT8（有符号）量化格式
 * 适用于 RKNPU2 等支持 INT8 量化推理的平台
 *
 * 参数: 与 process_u8 相同，只是数据类型为 int8_t
 * 返回: 有效检测框数量
 */
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // 将 FP32 阈值转换为 INT8 量化值
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score_sum 快速过滤低置信度区域
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            // 找到置信度最高的类别
            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // 执行边界框解码
            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                // YOLO 解码公式
                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

/**
 * process_fp32: 处理 FP32（32位浮点）格式的 YOLO 模型输出
 *
 * 适用于未量化的模型（FP32 模型）或不支持量化的平台
 * 由于不需要量化/反量化处理，代码更简单直接
 *
 * 参数:
 *   box_tensor: 边界框回归张量（FP32 格式）
 *   score_tensor: 类别置信度张量（FP32 格式）
 *   score_sum_tensor: 置信度求和张量（FP32 格式，可选）
 *   grid_h, grid_w: 特征图尺寸
 *   stride: 检测头步长
 *   dfl_len: DFL 分布区间长度
 *   boxes: 输出参数，存储边界框坐标
 *   objProbs: 输出参数，存储置信度分数
 *   classId: 输出参数，存储类别 ID
 *   threshold: 置信度阈值
 *
 * 返回:
 *   validCount: 有效检测框数量
 */
static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // score_sum 快速过滤
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            // 找到最高置信度类别
            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // 解码边界框
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                // FP32 格式不需要反量化，直接读取
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                // YOLO 解码公式
                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);  // FP32 直接使用，无需反量化
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}


#if defined(RV1106_1103)
/**
 * process_i8_rv1106: RV1106/RV1103 平台专用 INT8 处理函数
 *
 * 针对 RV1106/RV1103 平台的数据布局进行了优化
 * 数据布局说明：
 *   - score_tensor: [1, 80, 80, 80] NHWC 格式，即 (h, w, c) 排列
 *                   每个像素点 80 个类别分数连续存放
 *   - box_tensor: [1, 80, 80, 320] NHWC 格式，即 (h, w, 4*dfl_len) 排列
 *                   每个像素点 4*dfl_len 个回归值连续存放
 *
 * 与通用 process_i8 的主要区别：
 *   1. 数据寻址方式针对 NHWC 布局优化
 *   2. 每个像素点的多通道数据是连续存储的
 *
 * 参数:
 *   box_tensor: 边界框回归张量（INT8 NHWC 格式）
 *   其他参数同 process_i8
 *
 * 返回:
 *   validCount: 有效检测框数量
 */
static int process_i8_rv1106(int8_t *box_tensor, int32_t box_zp, float box_scale,
                             int8_t *score_tensor, int32_t score_zp, float score_scale,
                             int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                             int grid_h, int grid_w, int stride, int dfl_len,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId,
                             float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // 阈值量化
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // score_sum 快速过滤
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            // NHWC 布局：当前像素的类别分数连续存储
            int8_t max_score = -score_zp;
            offset = offset * OBJ_CLASS_NUM;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset + c] > score_thres_i8) && (score_tensor[offset + c] > max_score)) {
                    max_score = score_tensor[offset + c];
                    max_class_id = c;
                }
            }

            // 解码边界框
            if (max_score > score_thres_i8) {
                // NHWC 布局：当前像素的回归值连续存储
                offset = (i * grid_w + j) * 4 * dfl_len;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k], box_zp, box_scale);
                }
                compute_dfl(before_dfl, dfl_len, box);

                // YOLO 解码公式
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    printf("validCount=%d\n", validCount);
    printf("grid h-%d, w-%d, stride %d\n", grid_h, grid_w, stride);
    return validCount;
}
#endif

/**
 * post_process: YOLO 模型后处理主函数
 *
 * 这是整个后处理模块的入口函数，负责：
 *   1. 自动识别 YOLO26 vs YOLO8 模型（根据输出通道数）
 *   2. 根据不同平台（RV1106/RV1103/其他）选择合适的解码函数
 *   3. 执行 NMS 去重
 *   4. 将结果转换为统一的输出格式
 *
 * 参数:
 *   app_ctx: RKNN 应用上下文，包含模型配置信息
 *   outputs: 模型输出数据（rknn_tensor_mem* 数组或 rknn_output 数组）
 *   letter_box: letterbox 变换信息，用于还原到原图坐标
 *   conf_threshold: 置信度阈值
 *   nms_threshold: NMS 阈值
 *   od_results: 输出参数，存储最终检测结果
 *
 * 返回:
 *   0 = 成功
 */
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
#if defined(RV1106_1103) 
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    // default 3 branch
#ifdef RKNPU1
    int dfl_len = app_ctx->output_attrs[0].dims[2] / 4;
#else
    int dfl_len = app_ctx->output_attrs[0].dims[1] /4;
#endif
    int output_per_branch = app_ctx->io_num.n_output / 3;

    /* YOLO26 自动识别：3 个 head，每个 head 有 reg/cls 两个输出，且回归通道数为 4（非 DFL） */
    int box_c = 0;
#if defined(RV1106_1103)
    box_c = app_ctx->output_attrs[0].dims[3];
#else
    box_c = app_ctx->output_attrs[0].dims[1];
#endif
    int is_yolo26 = 0;
    if (app_ctx->is_quant && app_ctx->io_num.n_output == 6 && output_per_branch == 2)
    {
        int ok = 1;
        for (int head = 0; head < 3; head++)
        {
            int reg_idx = head * 2 + 0;
            int cls_idx = head * 2 + 1;
#if defined(RV1106_1103)
            int reg_c = app_ctx->output_attrs[reg_idx].dims[3];
            int cls_c = app_ctx->output_attrs[cls_idx].dims[3];
#else
            int reg_c = app_ctx->output_attrs[reg_idx].dims[1];
            int cls_c = app_ctx->output_attrs[cls_idx].dims[1];
#endif
            if (reg_c != 4 || cls_c != OBJ_CLASS_NUM)
            {
                ok = 0;
                break;
            }
        }
        is_yolo26 = ok;
    }

    printf("post_process: n_output=%d output_per_branch=%d box_c=%d is_quant=%d => is_yolo26=%d\n",
           app_ctx->io_num.n_output, output_per_branch, box_c, (int)app_ctx->is_quant, is_yolo26);

#if defined(RV1106_1103)
    if (is_yolo26)
    {
        printf("post_process: detected YOLO26 layout (RV1106), use YOLO26 postprocess.\n");

        DetectRectYolo26 rects[OBJ_NUMB_MAX_SIZE];
        int rect_num = 0;

        // 按实际输出顺序解析：每个 head 2 个输出(reg/cls)，grid/stride 从 output_attrs 动态推导
        for (int head = 0; head < 3; head++)
        {
            int reg_idx = head * 2 + 0;
            int cls_idx = head * 2 + 1;

            int grid_h_ = app_ctx->output_attrs[reg_idx].dims[1];
            int grid_w_ = app_ctx->output_attrs[reg_idx].dims[2];
            int stride_ = model_in_h / grid_h_;

            printf("yolo26 head=%d reg_idx=%d cls_idx=%d grid=%dx%d stride=%d\n",
                   head, reg_idx, cls_idx, grid_h_, grid_w_, stride_);

            // RV1106 输出通常是 NHWC：每个像素点的通道连续存放
            yolo26_decode_head_nhwc((int8_t *)_outputs[reg_idx]->virt_addr,
                                    app_ctx->output_attrs[reg_idx].zp,
                                    app_ctx->output_attrs[reg_idx].scale,
                                    (int8_t *)_outputs[cls_idx]->virt_addr,
                                    app_ctx->output_attrs[cls_idx].zp,
                                    app_ctx->output_attrs[cls_idx].scale,
                                    grid_h_, grid_w_, stride_, conf_threshold,
                                    rects, OBJ_NUMB_MAX_SIZE, &rect_num);
        }

        int last_count = 0;
        od_results->count = 0;
        for (int i = 0; i < rect_num && last_count < OBJ_NUMB_MAX_SIZE; i++)
        {
            int classId = rects[i].classId;
            float obj_conf = rects[i].score;
            float x1 = rects[i].xmin - letter_box->x_pad;
            float y1 = rects[i].ymin - letter_box->y_pad;
            float x2 = rects[i].xmax - letter_box->x_pad;
            float y2 = rects[i].ymax - letter_box->y_pad;

            od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = classId;
            last_count++;
        }
        od_results->count = last_count;
        return 0;
    }
#else
    if (is_yolo26)
    {
        printf("post_process: detected YOLO26 layout, use YOLO26 postprocess.\n");

        DetectRectYolo26 rects[OBJ_NUMB_MAX_SIZE];
        int rect_num = 0;

        for (int head = 0; head < 3; head++)
        {
            int reg_idx = head * 2 + 0;
            int cls_idx = head * 2 + 1;

            int grid_h_ = app_ctx->output_attrs[reg_idx].dims[2];
            int grid_w_ = app_ctx->output_attrs[reg_idx].dims[3];
            int stride_ = model_in_h / grid_h_;

            // 非 RV1106 通常是 NCHW/CHW 方式访问更常见
            yolo26_decode_head_chw((int8_t *)_outputs[reg_idx].buf,
                                   app_ctx->output_attrs[reg_idx].zp,
                                   app_ctx->output_attrs[reg_idx].scale,
                                   (int8_t *)_outputs[cls_idx].buf,
                                   app_ctx->output_attrs[cls_idx].zp,
                                   app_ctx->output_attrs[cls_idx].scale,
                                   grid_h_, grid_w_, stride_, conf_threshold,
                                   rects, OBJ_NUMB_MAX_SIZE, &rect_num);
        }

        int last_count = 0;
        od_results->count = 0;
        for (int i = 0; i < rect_num && last_count < OBJ_NUMB_MAX_SIZE; i++)
        {
            int classId = rects[i].classId;
            float obj_conf = rects[i].score;
            float x1 = rects[i].xmin - letter_box->x_pad;
            float y1 = rects[i].ymin - letter_box->y_pad;
            float x2 = rects[i].xmax - letter_box->x_pad;
            float y2 = rects[i].ymax - letter_box->y_pad;

            od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = classId;
            last_count++;
        }
        od_results->count = last_count;
        return 0;
    }
#endif

    for (int i = 0; i < 3; i++)
    {
#if defined(RV1106_1103)
        dfl_len = app_ctx->output_attrs[0].dims[3] /4;
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3) {
            score_sum = _outputs[i * output_per_branch + 2]->virt_addr;
            score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[2];
        stride = model_in_h / grid_h;
        
        if (app_ctx->is_quant) {
            validCount += process_i8_rv1106((int8_t *)_outputs[box_idx]->virt_addr, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                (int8_t *)_outputs[score_idx]->virt_addr, app_ctx->output_attrs[score_idx].zp,
                                app_ctx->output_attrs[score_idx].scale, (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                grid_h, grid_w, stride, dfl_len, filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            printf("RV1106/1103 only support quantization mode\n", LABEL_NALE_TXT_PATH);
            return -1;
        }

#else
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

#ifdef RKNPU1
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[0];
#else
        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];
#endif
        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
#ifdef RKNPU1
            validCount += process_u8((uint8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (uint8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (uint8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
#else
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold);
#endif
        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
#endif
    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

/**
 * init_post_process: 初始化后处理模块
 *
 * 在模型推理之前调用，加载类别标签文件
 * 将 COCO 80 类别的名称读取到内存中
 *
 * 返回:
 *   0 = 成功
 *   -1 = 失败（文件打开失败等）
 */
int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("加载类别标签文件失败: %s\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

/**
 * coco_cls_to_name: 根据类别 ID 获取类别名称
 *
 * 参数:
 *   cls_id: 类别 ID（从 0 开始）
 *
 * 返回:
 *   对应的类别名称字符串，如果 ID 无效则返回 "null"
 */
char *coco_cls_to_name(int cls_id)
{
    // 检查类别 ID 是否在有效范围内
    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    // 返回对应索引的标签
    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

/**
 * deinit_post_process: 释放后处理模块资源
 *
 * 在程序结束前调用，释放通过 malloc 分配的标签内存
 * 防止内存泄漏
 */
void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
