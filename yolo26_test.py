import os
import cv2
import numpy as np
from rknn.api import RKNN
 
# ====================== 配置 ======================
ONNX_MODEL = 'yolo26n_last.onnx'
RKNN_MODEL = "yolo26n_last.rknn"
DATASET = './dataset.txt'  # 量化数据集
QUANTIZE_ON = True
IMG_SIZE = (640, 640)  # (width, height)
 
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
 
CLASS_NAME = ["person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "]
 
# ✅ 去除类别名尾随空格
CLASS_NAME = [name.strip() for name in CLASS_NAME]
 
# ====================== 工具函数 ======================
def letterbox_resize(image, size, bg_color=114):
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy
 
def sigmoid(x):
    # 防止溢出
    return 1 / (1 + np.exp(-np.clip(x, -88.72, 88.72)))
 
def softmax(x, axis):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
 
def dfl(x):
    """
    Distribution Focal Loss (DFL) in pure NumPy.
    Input: x of shape [16, H, W] or [16, N]
    Output: [4, H, W] or [4, N]
    """
    assert x.shape[0] == 16, f"DFL expects 16 channels, got {x.shape[0]}"
    x = x.reshape(4, 4, -1)  # [4 coords, 4 bins, N]
    x = softmax(x, axis=1)   # softmax over the 4 bins
    acc = np.arange(4, dtype=np.float32).reshape(1, 4, 1)  # [1, 4, 1]
    x = np.sum(x * acc, axis=1)  # weighted sum → [4, N]
    return x
 
# ====================== NMS ======================
def nms(boxes, scores, thresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
 
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
 
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
 
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
 
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
 
        order = order[1:][iou <= thresh]
 
    return keep
 
# ====================== YOLOv8 后处理（官方公式）======================
def post_process(outputs, scale, dx, dy):
    boxes_list, scores_list, classes_list = [], [], []
    strides = [8, 16, 32]
 
    for i in range(3):
        reg = outputs[i * 2 + 0][0]  # [4, H, W] ← 确认是这个 shape
        cls = outputs[i * 2 + 1][0]  # [num_classes, H, W]
 
        _, H, W = reg.shape
        stride = strides[i]
 
        # 展平
        reg_flat = reg.reshape(4, -1)  # [4, N]
        cls_flat = cls.reshape(cls.shape[0], -1).T  # [N, num_classes]
 
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32).flatten()  # [N]
        grid_y = grid_y.astype(np.float32).flatten()  # [N]
 
        # ⭐⭐⭐ 关键：完全照搬 C++ 公式 ⭐⭐⭐
        tx = reg_flat[0]  # 注意：C++ 里是 -box[0]
        ty = reg_flat[1]  #        -box[1]
        tw = reg_flat[2]  #        +box[2]
        th = reg_flat[3]  #        +box[3]
 
        x1 = (-tx + grid_x + 0.5) * stride
        y1 = (-ty + grid_y + 0.5) * stride
        x2 = (tw + grid_x + 0.5) * stride
        y2 = (th + grid_y + 0.5) * stride
 
        w = x2 - x1
        h = y2 - y1
 
        # 过滤极小框（可选）
        # valid_wh = (w > 1e-4) & (h > 1e-4)
        # if not np.any(valid_wh): continue
 
        boxes = np.stack([x1, y1, w, h], axis=1)  # [N, 4] → (x1, y1, w, h)
 
        # 分类
        cls_prob = sigmoid(cls_flat)
        scores = np.max(cls_prob, axis=1)
        class_ids = np.argmax(cls_prob, axis=1)
 
        valid_mask = scores >= OBJ_THRESH
        if not np.any(valid_mask):
            continue
 
        boxes_v = boxes[valid_mask]
        scores_v = scores[valid_mask]
        classes_v = class_ids[valid_mask]
 
        # 转为 (x1, y1, x2, y2) 用于 NMS
        boxes_xyxy = np.copy(boxes_v)
        boxes_xyxy[:, 2] = boxes_v[:, 0] + boxes_v[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_v[:, 1] + boxes_v[:, 3]  # y2 = y1 + h
 
        # 还原到原图
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dx) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dy) / scale
 
        boxes_list.append(boxes_xyxy)
        scores_list.append(scores_v)
        classes_list.append(classes_v)
 
    if not boxes_list:
        return None, None, None
 
    boxes_all = np.concatenate(boxes_list, axis=0)
    scores_all = np.concatenate(scores_list, axis=0)
    classes_all = np.concatenate(classes_list, axis=0)
 
    keep = nms(boxes_all, scores_all, NMS_THRESH)
    if len(keep) == 0:
        return None, None, None
 
    return boxes_all[keep], classes_all[keep], scores_all[keep]

# ====================== RKNN 初始化/仿真 ======================
def get_rknn(simulate=True):
    rknn = RKNN(verbose=True)
    print('--> Config model')
    platform = 'rv1103'
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rv1103')
    print('done')
    if simulate:
        print("[仿真] 加载 ONNX 并 build...")
        ret = rknn.load_onnx(model=ONNX_MODEL)
        if ret != 0:
            raise RuntimeError(f"[仿真] load_onnx 失败：{ret}")
        ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        if ret != 0:
            raise RuntimeError(f"[仿真] build 失败：{ret}")
        
        # ⭐⭐⭐ 新增：导出 RKNN 模型文件 ⭐⭐⭐
        print(f"[仿真] 导出 RKNN 模型到 {RKNN_MODEL}...")
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"[仿真] export_rknn 失败：{ret}")
        print(f"[仿真] RKNN 模型已保存：{RKNN_MODEL}")
        
        ret = rknn.init_runtime(target=platform, device_id='eba42d647fb3dde0')
        if ret != 0:
            raise RuntimeError(f"[仿真] init_runtime 失败：{ret}")
        print("[仿真] ONNX->RKNN 仿真环境就绪")
    else:
        print("[板子] 加载 RKNN 模型...")
        ret = rknn.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f"[板子] load_rknn 失败：{ret}")
        ret = rknn.init_runtime()  # 板子推理时可指定 target/device_id
        if ret != 0:
            raise RuntimeError(f"[板子] init_runtime 失败：{ret}")
        print("[板子] RKNN 模型环境就绪")
    return rknn
 
# ====================== 推理接口 ======================
def detect_objects(img, rknn, return_vis=False):
    img_r, scale, dx, dy = letterbox_resize(img, IMG_SIZE)
    input_data = np.expand_dims(img_r, 0)
    outputs = rknn.inference(inputs=[input_data])

    # 打印输出形状（仅首次）
    if not hasattr(detect_objects, '_printed'):
        print(f"\n>>> 输出数量: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    output[{i}].shape = {out.shape}")
        detect_objects._printed = True

    boxes, cls_ids, scores = post_process(outputs, scale, dx, dy)

    if boxes is None or len(scores) == 0:
        if return_vis:
            return [], [], [], img.copy()
        return [], [], []

    if return_vis:
        vis = img.copy()
        h_img, w_img = vis.shape[:2]
        for box, cls_id, conf in zip(boxes, cls_ids, scores):
            x1, y1, x2, y2 = box
            x1 = int(np.clip(x1, 0, w_img))
            y1 = int(np.clip(y1, 0, h_img))
            x2 = int(np.clip(x2, 0, w_img))
            y2 = int(np.clip(y2, 0, h_img))

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
        return boxes.tolist(), cls_ids.tolist(), scores.tolist(), vis

    return boxes.tolist(), cls_ids.tolist(), scores.tolist()
 
# ====================== 测试 ======================
if __name__ == "__main__":
    IMG_PATH = "bus.jpg"
    OUTPUT_DIR = "./result"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {IMG_PATH}")

    # 选择模式：True=PC仿真，False=板子推理
    simulate = True
    rknn = get_rknn(simulate=simulate)

    boxes, cls_ids, scores, vis = detect_objects(img, rknn, return_vis=True)

    if len(scores) == 0:
        print("未检测到目标")
    else:
        print(f"检测到 {len(scores)} 个目标:")
        for i, (cls_id, conf) in enumerate(zip(cls_ids, scores)):
            cls_name = CLASS_NAME[int(cls_id)]
            print(f"  [{i+1}] 类别: {cls_name}, 置信度: {conf:.4f}")

        save_path = os.path.join(OUTPUT_DIR, "yolo26_result.jpg")
        cv2.imwrite(save_path, vis)
        print(f"可视化结果已保存: {save_path}")

    rknn.release()
