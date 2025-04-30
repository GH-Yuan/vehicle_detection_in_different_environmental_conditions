# utils/image_detector.py
import torch
from pathlib import Path
import logging
import cv2
import numpy as np
import sys # 导入 sys 模块用于检查模块导入状态

# 获取 logger 实例 - 移到文件顶部，确保在使用前初始化
logger = logging.getLogger(__name__)

# 导入 ultralytics 的 YOLO 类，用于加载模型 (包括 TensorRT engine)
# 在导入前先定义一个占位符类
_PlaceholderYOLO = None # 用于存储占位符 YOLO 类的引用
try:
    from ultralytics import YOLO
    logger.info("成功导入 ultralytics YOLO。")
    # 检查导入的 YOLO 是否是占位符，如果不是，则 _PlaceholderYOLO 保持 None
    if 'ultralytics' not in sys.modules or not hasattr(sys.modules['ultralytics'], 'YOLO'):
         # 如果 ultralytics 不在 sys.modules 或者没有 YOLO 属性，说明导入失败，使用占位符
         raise ImportError("ultralytics.YOLO not found")

except ImportError:
    logger.warning("无法导入 ultralytics。YOLO 模型将无法加载。请安装 ultralytics (pip install ultralytics)。")
    # 定义一个占位符 YOLO 类，防止 NameError
    class YOLO:
        def __init__(self, model_path):
            logger.error(f"ultralytics 未安装，无法加载 YOLO 模型: {model_path}")
            self.model_type = 'placeholder_yolo'
        def predict(self, image, conf=0.2, verbose=False):
            logger.error("占位符 YOLO 模型无法进行预测。")
            # 返回模拟的空结果
            class PlaceholderBoxes:
                 def __len__(self): return 0
            class PlaceholderResults:
                 def __init__(self):
                     self.boxes = PlaceholderBoxes()
                     self.speed = {} # 模拟空的 speed 字典
                 def plot(self): return np.zeros((100, 100, 3), dtype=np.uint8) # 返回黑色图像
            return [PlaceholderResults()]
    _PlaceholderYOLO = YOLO # 存储占位符 YOLO 类的引用


import time # 导入 time 用于测量性能

# 导入 SSD 相关的函数和类别名称
# 确保 SSD_CLASS_NAMES 在 ssd_detector 模块中已定义
from .ssd_detector import load_ssd_model, predict_ssd_image, draw_ssd_detections, SSD_CLASS_NAMES


def load_model(model_path: Path):
    """
    根据路径加载模型。支持 PyTorch (.pt), TensorRT (.engine) 和 SSD PyTorch (.pth) 格式。

    Args:
        model_path (Path): 模型文件路径。

    Returns:
        object or None: 加载的模型对象 (可以是 ultralytics.YOLO 或 torch.nn.Module)，如果失败则为 None。
    """
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return None

    try:
        logger.info(f"正在加载模型: {model_path}")

        # 根据文件扩展名或模型名称判断模型类型
        # 优先判断是否为 SSD PyTorch 模型 (.pth 且名称包含 'ssd')
        if model_path.suffix == '.pth' and 'ssd' in model_path.name.lower():
             # 尝试加载 SSD PyTorch 模型
             model = load_ssd_model(model_path)
             if model:
                 # 为 SSD 模型添加一个类型标识，方便后续判断
                 model.model_type = 'ssd'
                 logger.info("SSD 模型加载成功。")
                 return model
             else:
                 logger.error("加载 SSD 模型失败。")
                 return None

        elif model_path.suffix in ['.pt', '.engine']:
            # 尝试加载 YOLO PyTorch 或 TensorRT 模型 (使用 ultralytics)
            # ultralytics 的 YOLO 类可以直接加载多种格式的模型，包括 .pt 和 .engine
            # 它会自动检测文件类型并使用相应的后端

            # 检查 ultralytics 是否已成功导入且不是占位符 YOLO
            # 检查 sys.modules 中是否有 'ultralytics' 并且导入的 YOLO 不是我们定义的占位符类
            if 'ultralytics' not in sys.modules or YOLO is _PlaceholderYOLO:
                 logger.error("ultralytics 未安装或 YOLO 模型加载失败，无法加载 YOLO 模型。")
                 return None

            model = YOLO(str(model_path)) # YOLO 类接受字符串路径

            # 检查模型是否已加载到 GPU (对于 TensorRT engine，通常会自动在 GPU 上运行)
            if hasattr(model, 'device'):
                 logger.info(f"YOLO 模型加载成功。当前设备: {model.device}")
                 if str(model.device) == 'cpu':
                      logger.warning("⚠️ YOLO 模型加载到 CPU，TensorRT 模型应在 GPU 上运行。请检查您的 TensorRT 安装和环境。")
            else:
                 logger.info("YOLO 模型加载成功。设备信息不可用 (可能是 TensorRT engine)。")

            # 为 YOLO 模型添加一个类型标识
            model.model_type = 'yolo'
            return model

        else:
            logger.error(f"不支持的模型文件格式: {model_path.suffix} 或无法识别的模型名称。")
            return None

    except Exception as e:
        logger.error(f"加载模型失败: {model_path} - {str(e)}", exc_info=True)
        return None


def predict_image(image_path: Path, model_obj, output_folder: Path):
    """
    使用指定模型对象对图像进行预测。

    Args:
        image_path (Path): 输入图像文件路径。
        model_obj: 加载的模型对象 (ultralytics.YOLO 或 torch.nn.Module)。
        output_folder (Path): 输出图像保存目录。

    Returns:
        tuple: (输出图像路径, 性能指标字典)。如果预测失败，返回 (None, None)。
    """
    if model_obj is None:
        logger.error("模型对象为空，无法进行预测。")
        return None, None

    if not image_path.exists():
        logger.error(f"图像文件不存在: {image_path}")
        return None, None

    try:
        logger.info(f"正在对图像进行预测: {image_path}")

        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图像文件: {image_path}")
            return None, None

        # 根据模型类型执行预测和后处理
        model_type = getattr(model_obj, 'model_type', 'unknown')
        performance = {} # 初始化性能指标字典

        if model_type == 'ssd':
            logger.info("使用 SSD 模型进行预测。")
            # 使用 SSD 模型的预测逻辑
            # predict_ssd_image 内部会处理预测和性能指标
            detections, performance = predict_ssd_image(model_obj, image)
            logger.info("SSD 预测执行成功。")

            # 绘制检测框
            # 如果 predict_ssd_image 返回 None 或空列表，则不绘制
            if detections:
                annotated_frame = draw_ssd_detections(image, detections, class_names=SSD_CLASS_NAMES)
            else:
                annotated_frame = image # 没有检测结果时使用原帧
                logger.warning("SSD 图像检测未找到任何结果。")

        elif model_type == 'yolo':
            logger.info("使用 YOLO 模型进行预测。")
            # 使用 YOLO 模型的预测逻辑 (ultralytics)
            # 检查 ultralytics 是否已成功导入且不是占位符 YOLO
            if 'ultralytics' not in sys.modules or isinstance(model_obj, type) and model_obj is _PlaceholderYOLO:
                 logger.error("ultralytics 未安装或 YOLO 模型加载失败，无法进行 YOLO 预测。")
                 # 返回原始图像和空的性能指标
                 return image_path, {
                     'preprocess': 0.0,
                     'inference': 0.0,
                     'postprocess': 0.0,
                     'detections': 0
                 }

            start_predict = time.perf_counter()
            # predict 方法返回 Results 对象列表
            results = model_obj.predict(image, conf=0.2, verbose=False) # 直接对 NumPy 数组进行预测
            end_predict = time.perf_counter()
            predict_duration = (end_predict - start_predict) * 1000 # 总预测时间 (包含预处理、推理、后处理)

            # 提取性能指标 (ultralytics 提供的速度信息)
            speed = results[0].speed if results and len(results) > 0 and hasattr(results[0], 'speed') else {}
            detections_count = len(results[0].boxes) if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None else 0

            # 性能指标字典 (从 ultralytics speed 中提取或估算)
            performance = {
                 # ultralytics speed 字典通常包含 'preprocess', 'inference', 'postprocess'
                 'preprocess': speed.get('preprocess', 0),
                 'inference': speed.get('inference', 0),
                 'postprocess': speed.get('postprocess', 0),
                 'detections': detections_count,
                 # 如果需要总预测时间，可以使用 predict_duration
                 # 'total_predict': predict_duration
            }
            logger.info("YOLO 预测执行成功。")

            # 绘制检测框 (使用 ultralytics 的 plot 方法)
            if results and len(results) > 0:
                 annotated_frame = results[0].plot() # results[0].plot() 返回绘制了检测框的图像 (NumPy 数组)
            else:
                 annotated_frame = image # 如果没有检测结果，使用原帧
                 logger.warning("YOLO 图像检测未找到任何结果。")


        else:
            logger.error(f"无法识别的模型类型 '{model_type}'，无法进行预测。")
            # 返回原始图像和空的性能指标
            return image_path, {
                'preprocess': 0.0,
                'inference': 0.0,
                'postprocess': 0.0,
                'detections': 0
            }


        # 保存处理后的图像到指定的输出文件夹
        output_image_path = output_folder / 'output_image.jpg' # 为图像检测使用不同的文件名
        try:
            output_folder.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
            cv2.imwrite(str(output_image_path), annotated_frame) # 手动保存绘制后的图像
            logger.info(f"检测结果图像已保存到: {output_image_path}")
        except Exception as save_e:
             logger.error(f"保存检测结果图像失败: {str(save_e)}", exc_info=True)
             # 即使保存失败，也尝试返回原始图像路径和性能指标
             return image_path, performance


        # 返回输出图片的 Path 对象和性能指标
        # 注意：这里返回的是保存的图片路径，而不是原始图片路径
        return output_image_path, performance

    except Exception as e:
        logger.error(f"图像预测过程中发生错误: {str(e)}", exc_info=True)
        # 如果预测过程中发生错误，返回原始图像路径和空的性能指标
        return image_path, {
            'preprocess': 0.0,
            'inference': 0.0,
            'postprocess': 0.0,
            'detections': 0
        }
