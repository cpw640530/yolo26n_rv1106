"""
YOLO26 模型在 Rockchip RV1103/RV1106 平台上的推理测试脚本
功能：加载 RKNN 模型，对图像进行目标检测推理，并可视化检测结果
"""
import os
import cv2
import numpy as np
from rknn.api import RKNN

# ====================== 模型与推理配置 ======================
ONNX_MODEL = 'yolo26n_last.onnx'          # 原始 ONNX 模型路径
RKNN_MODEL = "yolo26n_last.rknn"          # 转换后的 RKNN 模型路径
DATASET = './dataset.txt'                 # 量化数据集路径（用于模型量化校准）
QUANTIZE_ON = True                        # 是否执行量化（True=INT8量化，False=fp32）
IMG_SIZE = (640, 640)                     # 模型输入尺寸 (width, height)

# ====================== 检测阈值配置 ======================
OBJ_THRESH = 0.25   # 目标置信度阈值：低于此值的检测结果将被过滤掉
NMS_THRESH = 0.45  # 非极大值抑制(NMS)阈值：用于去除重叠的重复检测框

# ====================== COCO 80 类别的名称列表 ======================
CLASS_NAME = ["person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "]

# 去除类别名称末尾的空格，避免显示时出现多余空白
CLASS_NAME = [name.strip() for name in CLASS_NAME]
 
# ====================== 工具函数 ======================

def letterbox_resize(image, size, bg_color=114):
    """
    Letterbox 缩放：保持图像宽高比，将图像缩放到目标尺寸并填充背景

    参数:
        image: 输入图像 (H, W, C)
        size: 目标尺寸 (target_w, target_h)
        bg_color: 填充背景颜色，默认为114（灰色）

    返回:
        canvas: 缩放并填充后的图像
        scale: 缩放比例
        dx: 图像在画布上的x方向偏移量
        dy: 图像在画布上的y方向偏移量
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    # 计算宽高缩放比例，取较小的比例以保证图像完整放入目标尺寸
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    # 先进行等比缩放
    resized = cv2.resize(image, (new_w, new_h))
    # 创建目标尺寸的画布，用背景色填充
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    # 计算图像在画布上的偏移量（居中放置）
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    # 将缩放后的图像放置到画布中央
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy

def sigmoid(x):
    """
    Sigmoid 激活函数：将数值压缩到 (0, 1) 区间

    参数:
        x: 输入值
    返回:
        Sigmoid 激活后的值，范围 (0, 1)
    """
    # 使用 np.clip 防止 exp 溢出：e^88.72 ≈ 1e38，再大会导致溢出
    return 1 / (1 + np.exp(-np.clip(x, -88.72, 88.72)))

def softmax(x, axis):
    """
    Softmax 函数：将输入转换为概率分布

    参数:
        x: 输入数组
        axis: 执行 softmax 的轴
    返回:
        归一化后的概率分布
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 数值稳定的 softmax 实现
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def dfl(x):
    """
    Distribution Focal Loss (DFL) 分布焦点损失函数 - 纯 NumPy 实现

    DFL 用于将离散的回归值转换为连续值，通过加权求和计算最终回归框坐标
    输入: x of shape [16, H, W] 或 [16, N]，16 = 4个坐标 × 4个分布区间
    输出: [4, H, W] 或 [4, N]，4个回归坐标值 (x, y, w, h)
    """
    assert x.shape[0] == 16, f"DFL expects 16 channels, got {x.shape[0]}"
    # 将数据重塑为 [4 coords, 4 bins, N]，4个坐标各对应4个分布区间
    x = x.reshape(4, 4, -1)
    # 对4个区间进行 softmax，得到各区间的概率分布
    x = softmax(x, axis=1)
    # 创建累积权重 [1, 4, 1]，代表区间索引 0, 1, 2, 3
    acc = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
    # 加权求和：sum(概率 × 区间索引) 得到最终连续回归值
    x = np.sum(x * acc, axis=1)
    return x
 
# ====================== NMS 非极大值抑制 ======================

def nms(boxes, scores, thresh):
    """
    非极大值抑制 (Non-Maximum Suppression)

    用于去除重叠的检测框，保留最优的检测结果。
    算法流程：
        1. 按置信度得分降序排列所有检测框
        2. 保留得分最高的框，然后移除与它 IoU > 阈值的所有框
        3. 重复上述过程直到处理完所有框

    参数:
        boxes: 检测框列表，格式为 [x1, y1, w, h]（左上角x, 左上角y, 宽度, 高度）
        scores: 每个框的置信度得分列表
        thresh: IoU 阈值，超过此值的框将被过滤掉

    返回:
        keep: 保留的检测框索引列表
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # 提取框的四个坐标 (x1, y1, x2, y2)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个框的面积
    areas = (x2 - x1) * (y2 - y1)
    # 按得分降序排列，argsort 返回升序索引，[::-1] 反转为降序
    order = scores.argsort()[::-1]

    keep = []  # 存储保留的框索引
    while order.size > 0:
        i = order[0]  # 得分最高的框
        keep.append(i)

        # 计算当前框与其余框的 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 重叠区域左上角 x
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 重叠区域左上角 y
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 重叠区域右下角 x
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 重叠区域右下角 y

        # 计算重叠区域面积，无重叠时为 0
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        # IoU = 重叠面积 / (两个框面积之和 - 重叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU <= 阈值的框（即重叠程度可接受的框）
        # 过滤掉与当前高得分框重叠严重的低得分框
        order = order[1:][iou <= thresh]

    return keep
 
# ====================== YOLO26 后处理（解码 + NMS）======================

def post_process(outputs, scale, dx, dy):
    """
    YOLO26 模型后处理函数

    YOLO26 的输出包含 3 个检测头（stride=8, 16, 32），每个头输出：
        - reg: 回归分支 [4, H, W]，包含边界框的偏移量 (tx, ty, tw, th)
        - cls: 分类分支 [num_classes, H, W]，包含每个类别的置信度

    参数:
        outputs: 模型输出的列表，每个元素对应一个分支的 tensor
        scale: letterbox 缩放比例，用于还原到原图坐标
        dx, dy: letterbox 填充偏移量，用于还原到原图坐标

    返回:
        boxes_all[keep]: 保留的检测框坐标 [x1, y1, x2, y2]
        classes_all[keep]: 保留的类别 ID
        scores_all[keep]: 保留的置信度得分
    """
    boxes_list, scores_list, classes_list = [], [], []
    strides = [8, 16, 32]  # 三个检测头的步长（特征图相对于原图的下采样倍数）

    for i in range(3):
        # 获取当前检测头的输出
        reg = outputs[i * 2 + 0][0]   # [4, H, W] 回归分支
        cls = outputs[i * 2 + 1][0]  # [num_classes, H, W] 分类分支

        _, H, W = reg.shape
        stride = strides[i]

        # 展平操作：将二维特征图展平为一维，便于批量处理
        reg_flat = reg.reshape(4, -1)  # [4, N]，N = H * W
        cls_flat = cls.reshape(cls.shape[0], -1).T  # [N, num_classes]

        # 生成网格坐标：记录每个特征点在其所属特征图中的位置
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32).flatten()  # [N]
        grid_y = grid_y.astype(np.float32).flatten()  # [N]

        # 关键：YOLO 官方解码公式（与 C++ 实现保持一致）
        # tx, ty: 中心点相对于特征图网格的偏移量（已激活，值范围约 -0.5 ~ 1.5）
        # tw, th: 宽高相对于锚框的指数增长（已激活，值范围约 0 ~ 4 左右）
        tx = reg_flat[0]  # 对应 C++ 中的 -box[0]
        ty = reg_flat[1]  # 对应 C++ 中的 -box[1]
        tw = reg_flat[2]  # 对应 C++ 中的 +box[2]
        th = reg_flat[3]  # 对应 C++ 中的 +box[3]

        # 解码得到边界框的绝对坐标（以输入尺寸 640x640 为基准）
        # 中心点坐标：(-tx + grid_x + 0.5) * stride 将偏移量转换为实际坐标
        x1 = (-tx + grid_x + 0.5) * stride
        y1 = (-ty + grid_y + 0.5) * stride
        x2 = (tw + grid_x + 0.5) * stride
        y2 = (th + grid_y + 0.5) * stride

        # 计算边界框的宽度和高度
        w = x2 - x1
        h = y2 - y1

        # 堆叠为边界框数组，格式为 (x1, y1, w, h)
        boxes = np.stack([x1, y1, w, h], axis=1)  # [N, 4]

        # 分类处理：计算每个检测点的类别概率
        cls_prob = sigmoid(cls_flat)  # 对所有类别分数应用 sigmoid
        scores = np.max(cls_prob, axis=1)  # 取每个检测点各类别中的最高概率作为置信度
        class_ids = np.argmax(cls_prob, axis=1)  # 获取最高概率对应的类别 ID

        # 根据置信度阈值过滤检测框
        valid_mask = scores >= OBJ_THRESH
        if not np.any(valid_mask):
            continue

        boxes_v = boxes[valid_mask]
        scores_v = scores[valid_mask]
        classes_v = class_ids[valid_mask]

        # 转换为 (x1, y1, x2, y2) 格式用于后续 NMS
        boxes_xyxy = np.copy(boxes_v)
        boxes_xyxy[:, 2] = boxes_v[:, 0] + boxes_v[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_v[:, 1] + boxes_v[:, 3]  # y2 = y1 + h

        # 还原到原图坐标：减去 letterbox 填充量，除以缩放比例
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dx) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dy) / scale

        boxes_list.append(boxes_xyxy)
        scores_list.append(scores_v)
        classes_list.append(classes_v)

    # 如果没有任何有效检测框
    if not boxes_list:
        return None, None, None

    # 合并三个检测头的结果
    boxes_all = np.concatenate(boxes_list, axis=0)
    scores_all = np.concatenate(scores_list, axis=0)
    classes_all = np.concatenate(classes_list, axis=0)

    # 执行 NMS，去除重叠的重复检测框
    keep = nms(boxes_all, scores_all, NMS_THRESH)
    if len(keep) == 0:
        return None, None, None

    # 返回 NMS 后的最终结果
    return boxes_all[keep], classes_all[keep], scores_all[keep]

# ====================== RKNN 模型加载与初始化 ======================

def get_rknn(simulate=True):
    """
    初始化 RKNN 推理上下文

    该函数负责：
        1. 配置模型输入的归一化参数（mean 和 std）
        2. 加载 ONNX 模型并转换为 RKNN 格式（或直接加载已有的 RKNN 模型）
        3. 初始化运行时环境

    参数:
        simulate: True = 在 PC 上通过 RKNN Toolkit 仿真运行（需要 ONNX 模型）
                  False = 在 RV1103/RV1106 开发板上运行（需要 RKNN 模型文件）

    返回:
        rknn: RKNN 推理上下文对象
    """
    rknn = RKNN(verbose=True)  # verbose=True 会在控制台输出详细的转换/推理日志
    print('--> 配置模型输入参数')

    # 设置模型的目标平台
    platform = 'rv1103'

    # 配置输入数据的归一化参数：
    # mean_values: RGB 三个通道的均值，默认为 [0, 0, 0]（不减去均值）
    # std_values: RGB 三个通道的标准差，默认为 [255, 255, 255]（除以 255 归一化到 0~1）
    # 这个配置与 ImageNet 数据的标准预处理方式一致
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rv1103')
    print('配置完成')

    if simulate:
        # ============== PC 仿真模式 ==============
        # 此模式使用 RKNN Toolkit 在 PC 上模拟 NPU 的推理行为
        # 适用于在没有真实开发板的情况下验证模型正确性

        print("[仿真] 加载 ONNX 模型并开始转换...")
        ret = rknn.load_onnx(model=ONNX_MODEL)
        if ret != 0:
            raise RuntimeError(f"[仿真] load_onnx 失败：{ret}")

        # 构建 RKNN 模型：do_quantization=True 会执行 INT8 量化
        # 量化需要 dataset.txt 中的图像数据进行校准，以减少精度损失
        ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        if ret != 0:
            raise RuntimeError(f"[仿真] build 失败：{ret}")

        # 导出 RKNN 模型文件到磁盘，方便后续直接加载使用
        print(f"[仿真] 导出 RKNN 模型到 {RKNN_MODEL}...")
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"[仿真] export_rknn 失败：{ret}")
        print(f"[仿真] RKNN 模型已保存：{RKNN_MODEL}")

        # 初始化 PC 端的推理运行时环境
        # target: 指定目标芯片平台
        # device_id: 指定具体的设备 ID（用于多设备场景）
        ret = rknn.init_runtime(target=platform, device_id='eba42d647fb3dde0')
        if ret != 0:
            raise RuntimeError(f"[仿真] init_runtime 失败：{ret}")
        print("[仿真] ONNX->RKNN 仿真环境就绪")

    else:
        # ============== 开发板运行模式 ==============
        # 直接加载预先转换好的 RKNN 模型文件
        # 这种方式跳过 ONNX 加载和转换步骤，启动更快

        print("[板子] 加载 RKNN 模型...")
        ret = rknn.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"[板子] load_rknn 失败：{ret}")

        # 初始化板子上的推理运行时
        # 在真实开发板上推理时，可以通过 target 和 device_id 指定具体设备
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"[板子] init_runtime 失败：{ret}")
        print("[板子] RKNN 模型环境就绪")

    return rknn
 
# ====================== 目标检测推理接口 ======================

def detect_objects(img, rknn, return_vis=False):
    """
    对输入图像执行目标检测

    参数:
        img: 输入图像（RGB 格式，BGR 格式均可）
        rknn: 已初始化的 RKNN 推理上下文
        return_vis: True = 返回带检测框的可视化图像，False = 不返回可视化结果

    返回:
        boxes: 检测框坐标列表 [x1, y1, x2, y2]，相对于原图尺寸
        cls_ids: 检测框对应的类别 ID 列表
        scores: 检测框对应的置信度得分列表
        vis: (可选) 带检测框的可视化图像，当 return_vis=True 时返回
    """
    # Step 1: Letterbox 缩放图像到模型输入尺寸
    # 返回：缩放后的图像、缩放比例、填充偏移量
    img_r, scale, dx, dy = letterbox_resize(img, IMG_SIZE)

    # Step 2: 准备模型输入
    # 增加 batch 维度：从 (H, W, C) 变为 (1, H, W, C)
    input_data = np.expand_dims(img_r, 0)

    # Step 3: 执行推理
    outputs = rknn.inference(inputs=[input_data])

    # Step 4: 打印输出形状（仅在首次调用时打印，方便调试）
    if not hasattr(detect_objects, '_printed'):
        print(f"\n>>> 模型输出数量: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    output[{i}].shape = {out.shape}")
        detect_objects._printed = True

    # Step 5: 后处理：解码模型输出，执行 NMS
    boxes, cls_ids, scores = post_process(outputs, scale, dx, dy)

    # Step 6: 处理无检测结果的情况
    if boxes is None or len(scores) == 0:
        if return_vis:
            return [], [], [], img.copy()
        return [], [], []

    # Step 7: 如果需要可视化，在图像上绘制检测框
    if return_vis:
        vis = img.copy()
        h_img, w_img = vis.shape[:2]
        for box, cls_id, conf in zip(boxes, cls_ids, scores):
            x1, y1, x2, y2 = box
            # 裁剪坐标到图像边界内，防止绘制出界
            x1 = int(np.clip(x1, 0, w_img))
            y1 = int(np.clip(y1, 0, h_img))
            x2 = int(np.clip(x2, 0, w_img))
            y2 = int(np.clip(y2, 0, h_img))

            # 获取类别名称
            cls_name = CLASS_NAME[int(cls_id)]

            # 绘制绿色边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 在检测框上方绘制类别名称和置信度
            cv2.putText(
                vis,
                f"{cls_name}:{conf:.2f}",
                (x1, max(y1 - 5, 0)),  # 文本位置在框上方 5 像素
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,                    # 字体大小
                (0, 255, 0),            # 绿色
                2                       # 文本粗细
            )
        return boxes.tolist(), cls_ids.tolist(), scores.tolist(), vis

    return boxes.tolist(), cls_ids.tolist(), scores.tolist()

# ====================== 主程序入口 ======================

if __name__ == "__main__":
    # 配置路径
    IMG_PATH = "bus.jpg"              # 输入图像路径
    OUTPUT_DIR = "./result"           # 检测结果输出目录

    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取输入图像
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {IMG_PATH}")

    # 选择运行模式：
    # True  = PC 仿真模式（使用 ONNX 模型转换后在 PC 上模拟推理）
    # False = 开发板模式（直接加载 RKNN 模型在 RV1103/RV1106 上推理）
    simulate = True

    # 初始化 RKNN 推理上下文
    rknn = get_rknn(simulate=simulate)

    # 执行目标检测
    boxes, cls_ids, scores, vis = detect_objects(img, rknn, return_vis=True)

    # 输出检测结果
    if len(scores) == 0:
        print("未检测到目标")
    else:
        print(f"检测到 {len(scores)} 个目标:")
        for i, (cls_id, conf) in enumerate(zip(cls_ids, scores)):
            cls_name = CLASS_NAME[int(cls_id)]
            print(f"  [{i+1}] 类别: {cls_name}, 置信度: {conf:.4f}")

        # 保存可视化结果
        save_path = os.path.join(OUTPUT_DIR, "yolo26_result.jpg")
        cv2.imwrite(save_path, vis)
        print(f"可视化结果已保存: {save_path}")

    # 释放 RKNN 资源
    rknn.release()
