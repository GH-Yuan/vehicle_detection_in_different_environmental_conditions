# utils/ssd_detector.py
import torch
import cv2
import numpy as np
import time
import logging
from pathlib import Path
# 导入 PIL 库的 Image 模块，用于图像预处理
from PIL import Image
import sys # 导入 sys 模块用于检查模块导入状态

# 导入 torchvision 的 transforms 模块，用于图像预处理
from torchvision import transforms


# 获取 logger 实例 - 移到文件顶部，确保在使用前初始化
logger = logging.getLogger(__name__)

# 导入 SSD 模型内部需要的组件
# 导入 SSD300_VGG16 模型构建函数
try:
    from torchvision.models.detection.ssd import ssd300_vgg16
    # 导入 SSD 预测头需要的组件
    from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
    # 导入权重枚举 (虽然我们加载自己的权重，但构建模型时可能需要)
    from torchvision.models.detection import SSD300_VGG16_Weights
    logger.info("成功导入 torchvision SSD 模型组件。")
except ImportError:
    logger.error("无法导入 torchvision SSD 模型组件。请安装 torchvision (pip install torchvision)。")
    # 定义占位符类和函数，防止 NameError，但模型将无法加载和工作
    class ssd300_vgg16(torch.nn.Module):
        def __init__(self, weights=None):
            super().__init__()
            logger.error("torchvision SSD 模型组件未安装，使用占位符模型。")
        def forward(self, x, targets=None):
            logger.error("占位符 SSD 模型无法进行前向传播。")
            # 返回模拟的空输出
            return [{'boxes': torch.empty((0, 4)), 'scores': torch.empty((0,)), 'labels': torch.empty((0,), dtype=torch.int64)}]

    class SSDClassificationHead(torch.nn.Module):
        def __init__(self, in_channels, num_anchors, num_classes):
             super().__init__()
             logger.error("torchvision SSD 模型组件未安装，使用占位符分类头。")
        def forward(self, x): return []

    class SSDRegressionHead(torch.nn.Module):
        def __init__(self, in_channels, num_anchors):
             super().__init__()
             logger.error("torchvision SSD 模型组件未安装，使用占位符回归头。")
        def forward(self, x): return []

    class SSD300_VGG16_Weights:
        COCO_V1 = None # 占位符权重


# 导入 torchvision 的 ops 模块用于 NMS
# 如果未安装 torchvision，请先安装： pip install torchvision
try:
    from torchvision.ops import nms
    logger.info("成功导入 torchvision.ops.nms")
except ImportError:
    logger.warning("无法导入 torchvision.ops.nms。请安装 torchvision (pip install torchvision) 以使用 NMS。")
    # 如果无法导入 NMS，定义一个占位符函数，但不执行实际 NMS
    def nms(boxes, scores, iou_threshold):
        logger.warning("torchvision.ops.nms 未导入，跳过 NMS。")
        # 返回所有索引，相当于不进行 NMS
        return torch.arange(boxes.shape[0], dtype=torch.long, device=boxes.device)


# ==============================================================================
# 根据您提供的数据集配置 data.yaml
# nc: 6
# names: ['car', 'bike', 'bus', 'truck', 'person', 'tractor']
# 定义类别名称和类别数量
# ==============================================================================
SSD_CLASS_NAMES = ['car', 'bike', 'bus', 'truck', 'person', 'tractor'] # 根据您提供的数据集类别名称进行设置
NUM_CLASSES_SSD = len(SSD_CLASS_NAMES) # 实际类别数量
NUM_CLASSES_MODEL = NUM_CLASSES_SSD + 1 # 模型总类别数 (含背景，SSD 通常将类别 0 保留给背景)


def load_ssd_model(model_path: Path):
    """
    加载 SSD PyTorch (.pth) 模型。

    Args:
        model_path (Path): 模型文件路径。

    Returns:
        torch.nn.Module or None: 加载的 SSD 模型对象，如果失败则为 None。
    """
    if not model_path.exists():
        logger.error(f"SSD 模型文件不存在: {model_path}")
        return None

    try:
        logger.info(f"正在加载 SSD 模型: {model_path}")

        # 检查是否有 GPU 可用，并设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"加载 SSD 模型到设备: {device}")

        # ======================================================================
        # 根据您提供的训练代码，构建 SSD300_VGG16 模型并替换预测头
        # ======================================================================
        # 1. 实例化预训练的 SSD300_VGG16 模型 (不加载 COCO 权重，因为我们要加载自己的权重)
        #    weights=None 表示不使用预训练权重初始化 Backbone
        model = ssd300_vgg16(weights=None) # 不在这里加载权重

        # 2. 获取预测头的输入通道数和每个特征图位置的锚框数量
        #    这些信息从构建的 ssd300_vgg16 模型中获取
        in_channels = [512, 1024, 512, 256, 256, 256] # SSD300_VGG16 的特征图通道数
        num_anchors = model.anchor_generator.num_anchors_per_location() # 每个特征图位置的锚框数量

        # 3. 创建新的分类和回归预测头，以匹配您的类别数 (含背景)
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=NUM_CLASSES_MODEL # 使用您的类别数 (含背景)
        )
        model.head.regression_head = SSDRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
        )
        logger.info(f"SSD 模型结构构建完成，总类别数 (含背景): {NUM_CLASSES_MODEL}")


        # 4. 加载您的模型权重 state_dict
        state_dict = torch.load(str(model_path), map_location=device) # 加载状态字典

        # 5. 加载状态字典到模型实例
        #    使用 strict=False 来尝试加载，即使 state_dict 和模型结构不完全匹配
        #    这会打印出 Unexpected key(s) 和 Missing key(s) 的警告信息
        #    如果您的模型结构与 ssd300_vgg16 并替换预测头完全一致，strict=True 应该也能工作
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False) # 返回加载结果
        if missing_keys:
            logger.warning(f"加载 state_dict 时缺少键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"加载 state_dict 时存在意外的键: {unexpected_keys}")

        # 6. 将模型移动到指定设备
        model.to(device)

        # 7. 设置模型为评估模式 model.eval()，关闭 dropout 和 batchnorm
        model.eval()

        logger.info("SSD 模型加载成功 (请检查上面的警告信息，确保模型结构匹配)。")

        return model

    except Exception as e:
        logger.error(f"加载 SSD 模型失败: {model_path} - {str(e)}", exc_info=True)
        return None

def preprocess_ssd_image(image: np.ndarray, target_size=(300, 300)):
    """
    SSD 模型的图像预处理。根据您提供的训练代码进行修改。

    Args:
        image (np.ndarray): 输入图像 (OpenCV 格式)。
        target_size (tuple): SSD 模型期望的输入尺寸 (宽, 高)。SSD300 通常是 (300, 300)。

    Returns:
        torch.Tensor: 预处理后的图像 Tensor。
    """
    # ==========================================================================
    # 根据您提供的训练代码中的 Transforms 进行修改
    # 包括： resizing, BGR to RGB, ToTensor, Normalization (mean/std)
    # ==========================================================================
    # logger.warning("请根据您的 SSD 模型训练时的预处理方式修改 preprocess_ssd_image 函数！") # 移除警告，因为已经根据用户代码修改

    # 转换为 RGB (如果您的模型期望 RGB 输入，通常是这样)
    # OpenCV 读取的是 BGR 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将 NumPy 数组转换为 PIL Image，以便应用 torchvision transforms
    # Image 类已在文件顶部导入
    image_pil = Image.fromarray(image_rgb)

    # 定义与训练时一致的 Transforms
    # 注意：这里只包含推理时需要的预处理，不包含数据增强
    # transforms 模块已在文件顶部导入
    preprocess_transforms = transforms.Compose([
        transforms.Resize(target_size), # Resize 到目标尺寸 (例如 300x300)
        transforms.ToTensor(), # 转换为 PyTorch Tensor (会自动将像素值从 [0, 255] 归一化到 [0, 1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

    # 应用 Transforms
    tensor_image = preprocess_transforms(image_pil)

    # 添加 batch 维度
    tensor_image = tensor_image.unsqueeze(0) # 添加 batch 维度

    return tensor_image

def postprocess_ssd_output(output, original_image_shape, conf_threshold=0.2, iou_threshold=0.45):
    """
    SSD 模型输出的后处理。

    Args:
        output: SSD 模型的原始输出。对于 torchvision 的 SSD 模型，这是一个字典列表，
                每个字典包含 'boxes' (Tensor [N, 4]), 'scores' (Tensor [N]), 'labels' (Tensor [N])。
        original_image_shape (tuple): 原始图像的形状 (高, 宽, 通道)。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): NMS 的 IoU 阈值。

    Returns:
        list: 检测结果列表，每个元素是一个字典 {'box': [x1, y1, x2, y2], 'score': s, 'class': c}。
    """
    # ==========================================================================
    # 根据 torchvision SSD 模型输出格式进行后处理
    # 包括：解析模型输出，应用置信度阈值，应用 NMS，坐标转换
    # ==========================================================================
    # logger.warning("请根据您的 SSD 模型输出格式和后处理逻辑修改 postprocess_ssd_output 函数！") # 移除警告，因为根据 torchvision SSD 输出修改

    detections = []
    if not output or len(output) == 0:
        logger.warning("模型输出为空，没有检测结果。")
        return detections # 返回空列表

    # torchvision 的检测模型输出是一个字典列表，通常只有一个元素 (对于单张图片)
    output_dict = output[0]

    if 'boxes' not in output_dict or 'scores' not in output_dict or 'labels' not in output_dict:
        logger.error("模型输出字典缺少 'boxes', 'scores' 或 'labels' 键。")
        return detections

    boxes = output_dict['boxes'] # Tensor [N, 4] (x1, y1, x2, y2) - 已经是像素坐标 (resized 尺寸)
    scores = output_dict['scores'] # Tensor [N]
    labels = output_dict['labels'] # Tensor [N] (类别 ID，0 通常是背景)

    logger.info(f"后处理前 - 原始检测数量: {boxes.shape[0]}")

    # 移除背景检测结果 (如果类别 0 是背景)
    # 并且只保留类别 ID 在有效范围内的结果
    valid_indices = (labels > 0) & (labels <= NUM_CLASSES_SSD) # 类别 ID > 0 且 <= 实际类别数量
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    logger.info(f"后处理后 - 移除背景和无效类别后检测数量: {boxes.shape[0]}")

    # 应用置信度阈值
    # 将默认置信度阈值从 0.5 降低到 0.3，以查看是否能捕获到更多检测
    confident_indices = scores > conf_threshold # 使用 conf_threshold 变量
    boxes = boxes[confident_indices]
    scores = scores[confident_indices]
    labels = labels[confident_indices]

    logger.info(f"后处理后 - 应用置信度阈值 ({conf_threshold:.2f}) 后检测数量: {boxes.shape[0]}")


    # 应用 NMS (非极大值抑制)
    if boxes.numel() > 0: # 只有当有检测框时才应用 NMS
        # nms 返回保留的检测结果的索引
        keep_indices = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        logger.info(f"后处理后 - 应用 NMS ({iou_threshold:.2f}) 后检测数量: {boxes.shape[0]}")
    else:
        logger.info("后处理后 - 没有检测框可应用 NMS。")


    # 将 boxes 坐标从 resized 尺寸转换回原始图像尺寸
    # 需要原始图像尺寸和 resized 图像尺寸
    original_h, original_w, _ = original_image_shape
    resized_h, resized_w = 300, 300 # SSD300 的固定输入尺寸

    # 计算缩放比例
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h

    # 缩放边界框 (如果 boxes 不为空)
    if boxes.numel() > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # 再次确认边界框没有越界 (因为浮点运算可能导致微小偏差)
        boxes[:, [0, 1]] = torch.clamp(boxes[:, [0, 1]], min=0)
        boxes[:, 2] = torch.clamp(boxes[:, 2], max=original_w * 1.0)
        boxes[:, 3] = torch.clamp(boxes[:, 3], max=original_h * 1.0)


    # 将结果转换为列表
    for i in range(boxes.shape[0]):
        detections.append({
            'box': boxes[i].tolist(), # [x1, y1, x2, y2]
            'score': scores[i].item(),
            'class': labels[i].item() - 1 # 将类别 ID 减 1，变回原始的 0-based 类别索引
        })

    logger.info(f"后处理完成，最终检测数量: {len(detections)}")
    return detections


def predict_ssd_image(model: torch.nn.Module, image: np.ndarray):
    """
    使用加载的 SSD 模型对图像进行预测。

    Args:
        model (torch.nn.Module): 加载的 SSD 模型对象。
        image (np.ndarray): 输入图像 (OpenCV 格式)。

    Returns:
        tuple: (检测结果列表, 性能指标字典)。如果预测失败，返回 ([], {})。
    """
    if model is None:
        logger.error("SSD 模型未加载，无法进行预测。")
        # 如果模型未加载，返回空列表和零性能指标
        return [], {
             'preprocess': 0.0,
             'inference': 0.0,
             'postprocess': 0.0,
             'detections': 0
        }

    try:
        # 预处理
        start_preprocess = time.perf_counter()
        processed_image = preprocess_ssd_image(image)
        end_preprocess = time.perf_counter()
        preprocess_time = (end_preprocess - start_preprocess) * 1000
        logger.info(f"SSD 预处理时间: {preprocess_time:.2f} ms")


        # 确保输入 Tensor 在正确的设备上
        device = next(model.parameters()).device # 获取模型所在的设备
        processed_image = processed_image.to(device)

        # 推理
        start_inference = time.perf_counter()
        with torch.no_grad(): # 在推理时禁用梯度计算
            # torchvision 的检测模型在 eval 模式下直接返回检测结果字典列表
            outputs = model(processed_image)
        end_inference = time.perf_counter()
        inference_time = (end_inference - start_inference) * 1000
        logger.info(f"SSD 推理时间: {inference_time:.2f} ms")


        # 后处理
        start_postprocess = time.perf_counter()
        # 将原始图像形状传递给后处理函数，用于坐标转换
        detections = postprocess_ssd_output(outputs, image.shape)
        end_postprocess = time.perf_counter()
        postprocess_time = (end_postprocess - start_postprocess) * 1000
        logger.info(f"SSD 后处理时间: {postprocess_time:.2f} ms")


        performance = {
             'preprocess': preprocess_time,
             'inference': inference_time,
             'postprocess': postprocess_time,
             'detections': len(detections)
        }
        logger.info(f"SSD 预测总检测数量: {performance['detections']}")


        return detections, performance

    except Exception as e:
        logger.error(f"SSD 图像预测过程中发生错误: {str(e)}", exc_info=True)
        # 如果预测过程中发生错误，返回空列表和零性能指标
        return [], {
             'preprocess': 0.0,
             'inference': 0.0,
             'postprocess': 0.0,
             'detections': 0
        }

def draw_ssd_detections(image: np.ndarray, detections: list, class_names=None):
    """
    在图像上绘制 SSD 检测结果。

    Args:
        image (np.ndarray): 输入图像 (OpenCV 格式)。
        detections (list): 检测结果列表，每个元素是一个字典 {'box': [x1, y1, x2, y2], 'score': s, 'class': c}。
        class_names (list): 类别名称列表，用于显示类别标签。

    Returns:
        np.ndarray: 绘制了检测框的图像。
    """
    annotated_image = image.copy()
    for det in detections:
        # 确保坐标是整数
        x1, y1, x2, y2 = [int(coord) for coord in det['box']]
        score = det['score']
        class_id = det['class'] # 这里是原始的 0-based 类别索引

        # 绘制矩形框
        color = (0, 255, 0) # 绿色框
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签和置信度
        # 使用 class_id 直接从 class_names 获取名称
        label = f"{class_names[class_id] if class_names and class_id < len(class_names) and class_id >= 0 else f'Class {class_id}'}: {score:.2f}"
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # 绘制文本背景框
        # 确保背景框不会超出图像顶部
        bg_y1 = max(0, y1 - text_height - baseline)
        cv2.rectangle(annotated_image, (x1, bg_y1), (x1 + text_width, y1), color, -1)
        # 绘制文本
        # 确保文本不会超出图像顶部
        text_y = max(text_height, y1 - baseline)
        cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # 黑色文本

    return annotated_image

# SSD 模型类别名称列表
# 根据您提供的数据集类别名称进行设置
SSD_CLASS_NAMES = ['car', 'bike', 'bus', 'truck', 'person', 'tractor']
# SSD 模型总类别数 (含背景)
NUM_CLASSES_SSD = len(SSD_CLASS_NAMES)
NUM_CLASSES_MODEL = NUM_CLASSES_SSD + 1 # 模型总类别数 (含背景，SSD 通常将类别 0 保留给背景)

