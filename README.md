# YOLO26模型在RV1103/RV1106平台的部署指南（详细版）

## 一、前言

RV1103/RV1106是瑞芯微推出的低功耗、高集成度的AIoT处理器，内置NPU算力单元，适用于边缘端轻量级目标检测场景。YOLO26作为YOLO系列的轻量化模型，兼顾检测速度与精度，非常适合在该类边缘平台部署。本文将详细讲解YOLO26模型从PT格式转换为RKNN格式，并完成RV1103/RV1106平台仿真测试与板端实际推理的全流程，包含环境搭建、模型转换、代码适配、测试验证等关键环节。

## 二、环境准备

### 2.1 基础环境要求

- 操作系统：Ubuntu 18.04/20.04/22.04（推荐，兼容性最佳）
- Python版本：3.8~3.10（需匹配RKNN-Toolkit2要求）
- 依赖工具：git、adb、cmake、交叉编译工具链（针对RV1103/RV1106）
- 硬件：RV1103/RV1106开发板（已烧录官方固件）、USB数据线（用于ADB连接）

### 2.2 Miniconda环境搭建

Miniconda可实现Python环境隔离，避免依赖冲突，步骤如下：

1. 下载并安装Miniconda：

   ```shell
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   # 安装完成后重启终端，使conda生效
   ```

2. 创建并激活专属环境：

   ```shell
   conda create -n yolo26_rknn python=3.10
   conda activate yolo26_rknn
   ```

## 三、模型转换前期准备

### 3.1 获取Ultralytics工程（YOLO26官方实现）

YOLO26基于Ultralytics框架开发，需先拉取工程并安装依赖：

```shell
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .  # 以可编辑模式安装，方便后续修改
pip install -r requirements.txt  # 补充安装框架依赖
cd ..
```

### 3.2 适配YOLO26转ONNX的工程修改

Ultralytics官方工程默认输出的ONNX模型可能不兼容RKNN-Toolkit2，需基于开源适配仓修改：

1. 拉取适配仓：

   ```shell
   git clone https://github.com/cqu20160901/yolo26_onnx_rknn.git
   ```

2. 核心修改说明：

   - 调整YOLO26的输出节点格式，适配RKNN的张量解析规则；
   - 移除ONNX模型中RKNN不支持的算子（如部分非标量常量算子）；
   - 确保模型输出通道与后处理逻辑匹配（YOLO26的3个检测尺度、80类分类输出）。

3. 将适配仓中的修改文件覆盖到Ultralytics工程对应目录（具体文件以适配仓说明为准）。

## 四、PT模型转ONNX模型

### 4.1 转换命令及参数说明

在激活的conda环境中，执行以下命令将YOLO26的PT权重转换为ONNX模型：

```shell
yolo export model=yolo26n.pt format=onnx opset=14 imgsz=640 simplify=True
```

关键参数解释：

- `model=yolo26n.pt`：指定待转换的YOLO26轻量化模型（也可替换为yolo26s.pt/yolo26m.pt）；
- `opset=14`：ONNX算子集版本，RKNN-Toolkit2对opset14兼容性最优；
- `imgsz=640`：模型输入尺寸（需与后续RKNN转换、板端推理保持一致）；
- `simplify=True`：简化ONNX模型，移除冗余节点，降低转换RKNN的报错概率。

### 4.2 转换验证

执行命令后，目录下会生成`yolo26n.onnx`文件，可通过以下方式验证模型有效性：

```shell
# 安装onnx工具
pip install onnx onnxruntime
# 验证模型结构
onnx check yolo26n.onnx
```

若提示“Valid ONNX model”，则说明ONNX模型生成正常。

## 五、ONNX模型转RKNN模型（含PC端仿真）

RKNN-Toolkit2是瑞芯微官方的模型转换工具，支持ONNX转RKNN，并提供PC端仿真功能（需连接开发板运行rknn_server）。

### 5.1 获取RKNN-Toolkit2工程

```shell
# 拉取官方仓（需匹配RV1103/RV1106版本）
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2
# 安装RKNN-Toolkit2依赖
pip install -r requirements_cp38_linux.txt  # 对应Python3.8，其他版本替换后缀
cd examples/onnx/yolov5  # 基于YOLOv5的示例目录改造
```

### 5.2 编写RKNN转换与仿真脚本

新建`yolo26_test.py`脚本，完整代码如下（含详细注释）：

```python
import os
import cv2
import numpy as np
from rknn.api import RKNN

# ====================== 核心配置项 ======================
ONNX_MODEL = 'yolo26n_last.onnx'  # 待转换的ONNX模型路径
RKNN_MODEL = "yolo26n_last.rknn"  # 输出的RKNN模型名称
DATASET = './dataset.txt'  # 量化数据集路径（每行一个图片路径，用于INT8量化）
QUANTIZE_ON = True  # 是否开启INT8量化（边缘端推荐开启，提升速度、降低内存）
IMG_SIZE = (640, 640)  # 模型输入尺寸（需与ONNX转换时一致）
OBJ_THRESH = 0.25  # 目标置信度阈值（过滤低置信度检测框）
NMS_THRESH = 0.45  # NMS非极大值抑制阈值（合并重叠框）
# COCO80类别名称（去除尾随空格，避免分类名匹配错误）
CLASS_NAME = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ====================== 图像预处理工具函数 ======================
def letterbox_resize(image, size, bg_color=114):
    """
    等比例缩放图像并填充黑边（保持目标比例，避免拉伸）
    :param image: 输入图像（HWC格式）
    :param size: 目标尺寸 (width, height)
    :param bg_color: 填充背景色
    :return: 处理后的图像、缩放比例、x/y方向填充偏移
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    # 计算等比例缩放因子
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    # 创建填充画布
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    dx = (target_w - new_w) // 2  # x方向填充偏移
    dy = (target_h - new_h) // 2  # y方向填充偏移
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy

# ====================== 激活函数与DFL解析 ======================
def sigmoid(x):
    """Sigmoid激活函数（防止数值溢出）"""
    return 1 / (1 + np.exp(-np.clip(x, -88.72, 88.72)))

def softmax(x, axis):
    """Softmax激活函数（用于DFL解析）"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 防止指数溢出
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def dfl(x):
    """
    解析Distribution Focal Loss（DFL）输出，还原边界框坐标
    :param x: DFL输出张量 [16, H, W] 或 [16, N]
    :return: 解析后的4个坐标偏移 [4, H, W] 或 [4, N]
    """
    assert x.shape[0] == 16, f"DFL需16通道输入，实际收到{x.shape[0]}通道"
    x = x.reshape(4, 4, -1)  # 拆分为4个坐标×4个bin×N个锚点
    x = softmax(x, axis=1)   # 对每个bin做Softmax
    acc = np.arange(4, dtype=np.float32).reshape(1, 4, 1)  # 权重系数 [1,4,1]
    x = np.sum(x * acc, axis=1)  # 加权求和得到最终坐标
    return x

# ====================== NMS非极大值抑制 ======================
def nms(boxes, scores, thresh):
    """
    非极大值抑制，去除重叠冗余框
    :param boxes: 检测框列表 [N,4] (x1,y1,x2,y2)
    :param scores: 置信度列表 [N]
    :param thresh: IOU阈值
    :return: 保留的检测框索引
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # 计算每个框的面积
    areas = (x2 - x1) * (y2 - y1)
    # 按置信度降序排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前框与剩余框的IOU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 交集面积
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        # IOU = 交集 / (当前框面积 + 剩余框面积 - 交集)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IOU小于阈值的框
        order = order[1:][iou <= thresh]
    return keep

# ====================== YOLO26后处理（匹配C++推理逻辑） ======================
def post_process(outputs, scale, dx, dy):
    """
    解析RKNN模型输出，还原检测框并过滤
    :param outputs: RKNN推理输出列表（6个张量，对应3个尺度的reg+cls）
    :param scale: 图像缩放比例
    :param dx/dy: 图像填充偏移
    :return: 过滤后的检测框、类别ID、置信度
    """
    boxes_list, scores_list, classes_list = [], [], []
    strides = [8, 16, 32]  # YOLO26的3个检测尺度步长

    for i in range(3):
        # 提取当前尺度的reg（边界框）和cls（分类）输出
        reg = outputs[i * 2 + 0][0]  # [4, H, W]
        cls = outputs[i * 2 + 1][0]  # [80, H, W]
        _, H, W = reg.shape
        stride = strides[i]

        # 展平张量便于计算
        reg_flat = reg.reshape(4, -1)  # [4, N] N=H*W
        cls_flat = cls.reshape(cls.shape[0], -1).T  # [N, 80]

        # 生成网格坐标（对应锚点中心）
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32).flatten()  # [N]
        grid_y = grid_y.astype(np.float32).flatten()  # [N]

        # 解析边界框（完全匹配板端C++逻辑）
        tx = reg_flat[0]
        ty = reg_flat[1]
        tw = reg_flat[2]
        th = reg_flat[3]
        # 还原绝对坐标
        x1 = (-tx + grid_x + 0.5) * stride
        y1 = (-ty + grid_y + 0.5) * stride
        x2 = (tw + grid_x + 0.5) * stride
        y2 = (th + grid_y + 0.5) * stride
        w = x2 - x1
        h = y2 - y1

        # 分类置信度计算（Sigmoid激活）
        cls_prob = sigmoid(cls_flat)
        scores = np.max(cls_prob, axis=1)  # 取每个框的最高类别置信度
        class_ids = np.argmax(cls_prob, axis=1)  # 取最高置信度对应的类别ID

        # 过滤低置信度框
        valid_mask = scores >= OBJ_THRESH
        if not np.any(valid_mask):
            continue

        # 提取有效框并转换为(x1,y1,x2,y2)格式（用于NMS）
        boxes_v = np.stack([x1, y1, w, h], axis=1)[valid_mask]
        boxes_xyxy = np.copy(boxes_v)
        boxes_xyxy[:, 2] = boxes_v[:, 0] + boxes_v[:, 2]
        boxes_xyxy[:, 3] = boxes_v[:, 1] + boxes_v[:, 3]

        # 还原到原始图像尺寸（抵消letterbox的填充和缩放）
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dx) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dy) / scale

        boxes_list.append(boxes_xyxy)
        scores_list.append(scores[valid_mask])
        classes_list.append(class_ids[valid_mask])

    # 合并所有尺度的结果
    if not boxes_list:
        return None, None, None
    boxes_all = np.concatenate(boxes_list, axis=0)
    scores_all = np.concatenate(scores_list, axis=0)
    classes_all = np.concatenate(classes_list, axis=0)

    # NMS去重
    keep = nms(boxes_all, scores_all, NMS_THRESH)
    if len(keep) == 0:
        return None, None, None

    return boxes_all[keep], classes_all[keep], scores_all[keep]

# ====================== RKNN模型初始化（仿真/板端） ======================
def get_rknn(simulate=True):
    """
    初始化RKNN模型（支持PC仿真和板端加载）
    :param simulate: True=PC端仿真（需连接开发板运行rknn_server），False=板端加载RKNN模型
    :return: 初始化后的RKNN实例
    """
    rknn = RKNN(verbose=True)  # 开启日志输出，便于调试

    # 配置模型参数（均值/方差用于归一化，目标平台指定RV1103）
    print('--> 配置RKNN模型参数')
    rknn.config(
        mean_values=[[0, 0, 0]],  # 图像归一化均值（与训练一致）
        std_values=[[255, 255, 255]],  # 图像归一化方差（与训练一致）
        target_platform='rv1103'  # RV1103/RV1106通用
    )
    print('模型参数配置完成')

    if simulate:
        # PC端仿真模式：加载ONNX并构建RKNN模型
        print("[仿真模式] 加载ONNX模型...")
        ret = rknn.load_onnx(model=ONNX_MODEL)
        if ret != 0:
            raise RuntimeError(f"ONNX模型加载失败，错误码：{ret}")

        print("[仿真模式] 构建RKNN模型（含量化）...")
        ret = rknn.build(
            do_quantization=QUANTIZE_ON,  # 开启INT8量化
            dataset=DATASET  # 量化数据集（需提前准备）
        )
        if ret != 0:
            raise RuntimeError(f"RKNN模型构建失败，错误码：{ret}")

        print(f"[仿真模式] 导出RKNN模型到 {RKNN_MODEL}...")
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"RKNN模型导出失败，错误码：{ret}")

        # 初始化仿真运行时（需ADB连接开发板，且开发板运行rknn_server）
        print("[仿真模式] 初始化运行时（连接开发板）...")
        ret = rknn.init_runtime(
            target='rv1103',
            device_id='eba42d647fb3dde0'  # 替换为实际的ADB设备ID（可通过adb devices查看）
        )
        if ret != 0:
            raise RuntimeError(f"运行时初始化失败，错误码：{ret}")
        print("[仿真模式] RKNN仿真环境就绪")
    else:
        # 板端模式：直接加载预生成的RKNN模型
        print("[板端模式] 加载RKNN模型...")
        ret = rknn.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"RKNN模型加载失败，错误码：{ret}")

        print("[板端模式] 初始化运行时...")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"运行时初始化失败，错误码：{ret}")
        print("[板端模式] RKNN模型环境就绪")
    return rknn

# ====================== 推理接口 ======================
def detect_objects(img, rknn, return_vis=False):
    """
    目标检测主接口
    :param img: 输入图像（HWC格式）
    :param rknn: RKNN实例
    :param return_vis: 是否返回可视化结果
    :return: 检测框、类别ID、置信度（+ 可视化图像，若return_vis=True）
    """
    # 图像预处理
    img_r, scale, dx, dy = letterbox_resize(img, IMG_SIZE)
    input_data = np.expand_dims(img_r, 0)  # 增加batch维度 [1,640,640,3]

    # RKNN推理
    outputs = rknn.inference(inputs=[input_data])

    # 首次运行打印输出张量形状（调试用）
    if not hasattr(detect_objects, '_printed'):
        print(f"\n>>> RKNN模型输出张量信息:")
        for i, out in enumerate(outputs):
            print(f"    output[{i}].shape = {out.shape}")
        detect_objects._printed = True

    # 后处理解析结果
    boxes, cls_ids, scores = post_process(outputs, scale, dx, dy)

    # 可视化结果（可选）
    if return_vis:
        vis = img.copy()
        h_img, w_img = vis.shape[:2]
        if boxes is not None and len(scores) > 0:
            for box, cls_id, conf in zip(boxes, cls_ids, scores):
                # 裁剪坐标到图像范围内
                x1 = int(np.clip(box[0], 0, w_img))
                y1 = int(np.clip(box[1], 0, h_img))
                x2 = int(np.clip(box[2], 0, w_img))
                y2 = int(np.clip(box[3], 0, h_img))
                # 绘制检测框和类别信息
                cls_name = CLASS_NAME[int(cls_id)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{cls_name}:{conf:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        return boxes.tolist() if boxes is not None else [], cls_ids.tolist() if cls_ids is not None else [], scores.tolist() if scores is not None else [], vis

    return boxes.tolist() if boxes is not None else [], cls_ids.tolist() if cls_ids is not None else [], scores.tolist() if scores is not None else []

# ====================== 测试主函数 ======================
if __name__ == "__main__":
    # 测试配置
    IMG_PATH = "bus.jpg"  # 测试图像路径（需提前放入当前目录）
    OUTPUT_DIR = "./result"
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建结果保存目录

    # 加载测试图像
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"测试图像未找到：{IMG_PATH}")

    # 选择运行模式：True=PC仿真，False=板端推理
    simulate = True
    rknn = get_rknn(simulate=simulate)

    # 执行检测
    boxes, cls_ids, scores, vis = detect_objects(img, rknn, return_vis=True)

    # 输出检测结果
    if len(scores) == 0:
        print("未检测到任何目标")
    else:
        print(f"检测到 {len(scores)} 个目标:")
        for i, (cls_id, conf) in enumerate(zip(cls_ids, scores)):
            cls_name = CLASS_NAME[int(cls_id)]
            print(f"  [{i+1}] 类别: {cls_name}, 置信度: {conf:.4f}")
        # 保存可视化结果
        save_path = os.path.join(OUTPUT_DIR, "yolo26_result.jpg")
        cv2.imwrite(save_path, vis)
        print(f"可视化结果已保存至: {save_path}")

    # 释放RKNN资源
    rknn.release()
```

### 5.3 量化数据集准备

INT8量化需提供数据集文件`dataset.txt`，格式为每行一个图像路径：

```
./dataset/0001.jpg
./dataset/0002.jpg
./dataset/0003.jpg
...
```

注：图像尺寸无需严格640×640，Toolkit会自动预处理。

### 5.4 仿真测试执行

#### 5.4.1 开发板端准备

1. 将开发板通过USB数据线连接至PC，确保ADB可识别：

   ```shell
   adb devices  # 应显示开发板设备ID
   (RKNN-Toolkit2) chenpengwen@4CE405CFCK:~/rknn/rknn-toolkit2-master/rknn-toolkit2/examples/onnx/yolov5$ adb devices
   List of devices attached
   eba42d647fb3dde0        device
   ```

2. 在开发板上启动rknn_server（需提前部署至板端）：

   ```shell
   # 板端执行
   ./rknn_server
   ```

#### 5.4.2 PC端执行仿真脚本

```shell
# 确保已激活conda环境
conda activate yolo26_rknn
# 执行转换与仿真脚本
python yolo26_test.py
```

#### 5.4.3 仿真结果验证

正常运行后会输出如下日志（关键信息）：

```
[仿真] 导出 RKNN 模型到 yolo26n_last.rknn...
[仿真] RKNN 模型已保存：yolo26n_last.rknn
I Connect to Device success!
D RKNNAPI: RKNN VERSION:
D RKNNAPI:   API: 2.3.2 (1842325 build@2025-03-30T09:55:23)
D RKNNAPI:   DRV: rknn_server: 2.3.2 (1842325 build@2025-03-30T09:54:54)

>>> RKNN模型输出张量信息:
    output[0].shape = (1, 4, 80, 80)
    output[1].shape = (1, 80, 80, 80)
    output[2].shape = (1, 4, 40, 40)
    output[3].shape = (1, 80, 40, 40)
    output[4].shape = (1, 4, 20, 20)
    output[5].shape = (1, 80, 20, 20)
检测到 5 个目标:
  [1] 类别: person, 置信度: 0.9021
  [2] 类别: person, 置信度: 0.9021
  [3] 类别: bus, 置信度: 0.9021
  [4] 类别: person, 置信度: 0.8642
  [5] 类别: person, 置信度: 0.5000
可视化结果已保存至: ./result/yolo26_result.jpg
```

同时会生成`yolo26n_last.rknn`模型文件和可视化结果图，说明仿真成功。

## 六、板端部署测试

### 6.1 获取RKNN Model Zoo工程

```shell
git clone https://github.com/rockchip-linux/rknn_model_zoo.git
cd rknn_model_zoo/examples/yolov8
```

### 6.2 板端后处理代码修改（适配YOLO26）

YOLO26的输出格式与YOLOv8略有差异，需修改`postprocess.cc`文件，核心修改如下：

#### 6.2.1 补充YOLO26专用解码函数

在`postprocess.cc`中添加YOLO26的边界框解码逻辑（适配RV1103/RV1106的NHWC输出格式）：

```cpp
/* ===================== YOLO26 专用后处理（纯 C 实现） ===================== */
typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRectYolo26;

static inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (uint32_t)(12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static float yolo26_sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

/* YOLO26 head 解码（NHWC排布，适配RV1103/RV1106） */
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
            int offset_hw = h * grid_w + w;

            // 查找最高置信度类别
            float cls_max = -1e9f;
            int cls_index = -1;
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

            // 置信度过滤
            float score = yolo26_sigmoid(DeQnt2F32((int8_t)cls_max, cls_zp, cls_scale));
            if (score <= threshold)
            {
                continue;
            }

            // 解析边界框
            int reg_base = offset_hw * 4;
            float cx = DeQnt2F32(reg[reg_base + 0], reg_zp, reg_scale);
            float cy = DeQnt2F32(reg[reg_base + 1], reg_zp, reg_scale);
            float cw = DeQnt2F32(reg[reg_base + 2], reg_zp, reg_scale);
            float ch = DeQnt2F32(reg[reg_base + 3], reg_zp, reg_scale);

            // 还原绝对坐标
            float center_x = (float)w + 0.5f;
            float center_y = (float)h + 0.5f;
            float xmin = (center_x - cx) * stride;
            float ymin = (center_y - cy) * stride;
            float xmax = (center_x + cw) * stride;
            float ymax = (center_y + ch) * stride;

            // 保存有效框
            if (*rect_count >= max_rects)
            {
                continue;
            }
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
```

#### 6.2.2 修改RV1106/RV1103的处理函数

适配YOLO26的输出张量解析逻辑，修改`process_i8_rv1106`函数：

```cpp
#if defined(RV1106_1103)
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
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过score_sum快速过滤低置信度框
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            // 查找最高置信度类别（适配NHWC格式的80类输出）
            int8_t max_score = -score_zp;
            offset = offset * OBJ_CLASS_NUM;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset + c] > score_thres_i8) && (score_tensor[offset + c] > max_score)) {
                    max_score = score_tensor[offset + c];
                    max_class_id = c;
                }
            }

            // 解析边界框（适配YOLO26的DFL输出）
            if (max_score > score_thres_i8) {
                offset = (i * grid_w + j) * 4 * dfl_len;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k], box_zp, box_scale);
                }
                // 解析DFL输出
                compute_dfl(before_dfl, dfl_len, box);
                
                // YOLO26坐标还原逻辑
                float x1 = (-box[0] + j + 0.5) * stride;
                float y1 = (-box[1] + i + 0.5) * stride;
                float x2 = (box[2] + j + 0.5) * stride;
                float y2 = (box[3] + i + 0.5) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                // 保存结果
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
#endif
```

### 6.3 板端编译与运行

#### 6.3.1 配置交叉编译环境

下载瑞芯微官方交叉编译工具链（`gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf`），并配置环境变量：

```shell
such as export GCC_COMPILER=~/opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf
```

#### 6.3.2 编译工程

```shell
chenpengwen@4CE405CFCK:~/rknn/rknn_model_zoo$ ./build-linux.sh -t rv1106 -a armhf -d yolov8
#生成rknn_model_zoo/install/rv1106_linux_armhf/rknn_yolov8_demo
```

#### 6.3.3 部署至开发板

1. 将编译生成的可执行文件、RKNN模型、类别标签文件拷贝至开发板：

   ```shell
   adb push rknn_yolov8_demo /userdata/
   ```

2. 板端执行推理：

   ```shell
   # 板端进入目录
   cd /userdata
   # 执行推理（替换为实际测试图像路径）
   ./yolov8_demo  yolo26n_last.rknn bus.jpg 
   ```

### 6.4 板端结果验证

正常运行后，板端会输出检测结果（类别、置信度、坐标），并生成可视化图像，示例输出：

```
load lable ./model/coco_80_labels_list.txt
检测到目标数：5
[1] 类别: person, 置信度: 0.90, 坐标: (120, 80, 180, 320)
[2] 类别: person, 置信度: 0.90, 坐标: (200, 90, 260, 330)
[3] 类别: bus, 置信度: 0.90, 坐标: (50, 60, 600, 400)
[4] 类别: person, 置信度: 0.86, 坐标: (300, 100, 360, 340)
[5] 类别: person, 置信度: 0.50, 坐标: (400, 110, 460, 350)
```

## 七、常见问题与排查

### 7.1 仿真模式ADB连接失败

- 现象：`adb: unable to connect for root: closed`
- 解决：
  1. 确认开发板已上电并通过USB连接PC；
  2. 重启开发板的ADB服务：`adb kill-server && adb start-server`；
  3. 确认开发板已开启ADB调试模式。

## 八、总结

本文详细讲解了YOLO26模型在RV1103/RV1106平台的全流程部署，核心步骤包括：环境搭建→PT转ONNX→ONNX转RKNN（PC仿真）→板端后处理适配→编译部署→验证。关键注意点：

1. 模型转换时需保证输入尺寸、算子集版本与平台兼容；
2. 板端后处理需严格匹配YOLO26的输出格式（尤其是RV1103/RV1106的NHWC排布）；
3. 仿真模式需确保ADB连接正常且开发板运行rknn_server。

通过以上步骤，可实现YOLO26模型在RV1103/RV1106平台的高效部署，兼顾检测精度与边缘端推理速度。
