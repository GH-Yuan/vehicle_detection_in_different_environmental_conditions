# utils/video_detector.py
import cv2
import time
from pathlib import Path
import logging
import numpy as np
import torch # 导入 torch 用于检查 CUDA 可用性

# 获取 logger 实例
logger = logging.getLogger(__name__)

# 从 image_detector 导入 load_model 函数
from .image_detector import load_model
# 导入 SSD 相关的预测和绘制函数
from .ssd_detector import predict_ssd_image, draw_ssd_detections, SSD_CLASS_NAMES # 导入 SSD_CLASS_NAMES


def process_video_detection(file, model_name, app_config):
    """
    处理视频文件上传、检测和结果保存。

    Args:
        file: 上传的视频文件对象。
        model_name: 使用的模型名称。
        app_config: Flask 应用的配置对象，包含路径信息。

    Returns:
        tuple: (处理后视频的URL, 错误消息, 性能指标字典)。
               如果成功，错误消息为 None，性能指标字典包含结果。
               如果失败，URL 为 None，错误消息包含错误信息，性能指标字典为空。
    """
    error_msg = None
    processed_video_url = None
    performance_metrics = {}

    upload_folder = Path(app_config['UPLOAD_FOLDER'])
    processed_video_folder = Path(app_config['PROCESSED_VIDEO_FOLDER'])
    model_config = app_config['MODEL_CONFIG']

    if model_name not in model_config:
        error_msg = f"无效的模型名称: {model_name}"
        logger.warning(error_msg)
        return None, error_msg, performance_metrics

    model_path_str = model_config.get(model_name)
    if not model_path_str or not Path(model_path_str).exists():
        error_msg = f"模型文件不存在: {model_path_str}"
        logger.error(error_msg)
        return None, error_msg, performance_metrics

    # 保存上传的视频文件
    filename = file.filename
    filename = Path(filename).name # 清理文件名
    upload_path = upload_folder / filename
    try:
        upload_folder.mkdir(parents=True, exist_ok=True)
        file.save(str(upload_path))
        logger.info(f"视频文件保存成功: {upload_path}")
    except Exception as e:
        error_msg = f"视频文件保存失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg, performance_metrics

    # 加载模型
    try:
        # 使用 image_detector 中的 load_model 函数加载模型
        # load_model 函数内部会处理 .pt, .engine 和 .pth 格式
        model = load_model(Path(model_path_str))
        if model is None:
             # load_model 内部已经记录了错误日志
             # 如果模型加载失败，直接返回错误
             # 注意：如果模型加载失败，后续的预测函数会返回零结果和零指标
             return None, f"加载模型失败: {model_path_str}", performance_metrics

        # 检查模型类型
        model_type = getattr(model, 'model_type', 'unknown')
        logger.info(f"加载的模型类型: {model_type}")

    except Exception as e:
        error_msg = f"加载模型失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg, performance_metrics

    # 处理视频
    # 固定处理后的视频文件名为 output.mp4
    processed_filename = 'output.mp4'
    processed_video_path = processed_video_folder / processed_filename

    cap = None
    out = None

    try:
        # 打开视频文件
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            error_msg = f"无法打开视频文件: {upload_path}"
            logger.error(error_msg)
            return None, error_msg, performance_metrics

        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 定义视频编码器和创建 VideoWriter 对象
        # 尝试使用 H.264 (avc1) 编码器，这是浏览器最广泛支持的格式
        # 如果 avc1 不可用，尝试其他的
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'), # H.264 - 最推荐，如果可用
            cv2.VideoWriter_fourcc(*'XVID'), # XVID - MPEG-4 ASP，兼容性尚可
            cv2.VideoWriter_fourcc(*'MJPG'), # MJPG - Motion JPEG，兼容性尚可，文件较大
            cv2.VideoWriter_fourcc(*'mp4v')  # mp4v - MPEG-4 Part 2，兼容性可能不如 H.264
        ]
        out = None
        for fourcc_code in fourcc_options:
            try:
                # 如果 output.mp4 已存在，VideoWriter 可能会覆盖它
                out = cv2.VideoWriter(str(processed_video_path), fourcc_code, fps, (frame_width, frame_height))
                if out is not None and out.isOpened():
                    logger.info(f"成功使用编码器 {fourcc_code} 创建 VideoWriter。")
                    break # 成功创建则跳出循环
                else:
                     logger.warning(f"尝试使用编码器 {fourcc_code} 创建 VideoWriter 失败。")
            except Exception as vw_e:
                 logger.warning(f"创建 VideoWriter 时发生异常，编码器 {fourcc_code}: {str(vw_e)}")

        if out is None or not out.isOpened():
             error_msg = "无法创建视频写入器，请检查系统是否支持常见的视频编码器（avc1, XVID, MJPG, mp4v）。"
             logger.error(error_msg)
             cap.release()
             return None, error_msg, performance_metrics

        logger.info("开始处理视频帧...")
        frame_count = 0
        total_inference_time = 0
        total_preprocess_time = 0
        total_postprocess_time = 0
        total_detections = 0
        video_processing_start_time = time.perf_counter() # 记录视频处理开始时间

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # 读取失败或视频结束

            frame_count += 1
            # 在帧上执行预测
            try:
                # TODO: 添加帧处理进度显示
                # 例如：logging.info(f"正在处理第 {frame_count}/{total_frames} 帧")

                # 根据模型类型调用相应的预测逻辑
                model_type = getattr(model, 'model_type', 'unknown')

                if model_type == 'ssd':
                    # 使用 SSD 模型的预测逻辑
                    # 注意：如果 SSD 模型加载或预测失败，predict_ssd_image 会返回 ([], {})
                    # 这将导致累加的性能指标和检测数量为 0
                    detections, performance = predict_ssd_image(model, frame)
                    # 如果 predict_ssd_image 返回 None 或空列表，则不绘制
                    if detections:
                        annotated_frame = draw_ssd_detections(frame, detections, class_names=SSD_CLASS_NAMES)
                    else:
                        annotated_frame = frame # 没有检测结果时使用原帧


                    # 累加性能指标 (SSD 预测函数返回的性能指标)
                    total_preprocess_time += performance.get('preprocess', 0)
                    total_inference_time += performance.get('inference', 0)
                    total_postprocess_time += performance.get('postprocess', 0)
                    total_detections += performance.get('detections', 0)

                elif model_type == 'yolo':
                    # 使用 YOLO 模型的预测逻辑 (ultralytics)
                    start_predict = time.perf_counter()
                    results = model.predict(frame, conf=0.2, verbose=False)
                    end_predict = time.perf_counter()
                    predict_duration = (end_predict - start_predict) * 1000

                    # 提取性能指标 (ultralytics speed)
                    speed = results[0].speed if results and len(results) > 0 and hasattr(results[0], 'speed') else {}
                    detections_count = len(results[0].boxes) if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None else 0

                    # 累加性能指标 (从 ultralytics speed 中提取或估算)
                    total_preprocess_time += speed.get('preprocess', 0)
                    total_inference_time += speed.get('inference', 0)
                    total_postprocess_time += speed.get('postprocess', 0)
                    total_detections += detections_count


                    # 绘制检测框 (使用 ultralytics 的 plot 方法)
                    if results and len(results) > 0:
                         annotated_frame = results[0].plot()
                    else:
                         annotated_frame = frame # 如果没有检测结果，使用原帧

                else:
                    logger.warning(f"不支持的模型类型 '{model_type}'，跳过帧处理。")
                    annotated_frame = frame # 使用原帧


                # 写入处理后的帧
                out.write(annotated_frame)

            except Exception as frame_e:
                logger.warning(f"处理视频帧 {frame_count} 时发生错误: {str(frame_e)}")
                # 可以选择跳过当前帧或停止处理
                pass # 这里选择跳过


        video_processing_end_time = time.perf_counter() # 记录视频处理结束时间
        total_video_processing_time = (video_processing_end_time - video_processing_start_time) * 1000 # 总视频处理时间（毫秒）


        logger.info(f"视频处理完成，共计处理 {frame_count} 帧")

        # 计算平均性能指标
        if frame_count > 0:
             performance_metrics = {
                 'average_preprocess': total_preprocess_time / frame_count,
                 'average_inference': total_inference_time / frame_count,
                 'average_postprocess': total_postprocess_time / frame_count,
                 'total_detections': total_detections,
                 'total_frames': frame_count,
                 'total_processing_time': total_video_processing_time # 添加总视频处理时间
             }
             logger.info(f"平均预处理时间: {performance_metrics['average_preprocess']:.2f} ms/帧")
             logger.info(f"平均推理时间: {performance_metrics['average_inference']:.2f} ms/帧")
             logger.info(f"平均后处理时间: {performance_metrics['average_postprocess']:.2f} ms/帧")
             logger.info(f"总检测目标数量: {performance_metrics['total_detections']}")
             logger.info(f"总帧数: {performance_metrics['total_frames']}")
             logger.info(f"总视频处理时间: {performance_metrics['total_processing_time']:.2f} ms")
        else:
             # 如果没有处理任何帧，性能指标保持为 0
             performance_metrics = {
                 'average_preprocess': 0.0,
                 'average_inference': 0.0,
                 'average_postprocess': 0.0,
                 'total_detections': 0,
                 'total_frames': 0,
                 'total_processing_time': 0.0
             }
             logger.warning("视频处理未完成任何帧，性能指标为 0。请检查视频文件或模型加载是否正常。")


        # 返回固定名称的视频URL
        processed_video_url = f'/static/output/{processed_filename}?t={int(time.time())}' # 添加时间戳防止缓存
        return processed_video_url, None, performance_metrics

    except Exception as e:
        error_msg = f"视频处理过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg, performance_metrics
    finally:
        # 确保在任何情况下都释放资源
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()
        # 添加一个短暂的延迟，确保文件句柄被释放
        time.sleep(0.1) # 延迟 100 毫秒
