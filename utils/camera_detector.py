# utils/camera_detector.py
import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
import torch # 导入 torch 用于检查 CUDA 可用性

# 获取 logger 实例
logger = logging.getLogger(__name__)

# 从 image_detector 导入 load_model 函数
from .image_detector import load_model
# 导入 SSD 相关的预测和绘制函数
# 确保 SSD_CLASS_NAMES 在 ssd_detector 模块中已定义
from .ssd_detector import predict_ssd_image, draw_ssd_detections, SSD_CLASS_NAMES


# 摄像头检测相关的全局变量
camera = None
output_frame = None # 用于存储最新编码帧的全局变量
lock = threading.Lock() # 用于保护输出帧的线程锁
model_for_camera = None # 摄像头检测使用的模型对象
camera_model_path = None # 摄像头模型路径 (Path 对象)
is_camera_running = False # 标记摄像头是否正在运行
current_camera_id = None # 当前打开的摄像头ID
current_model_name = None # 当前使用的模型名称
camera_thread = None # 初始化 camera_thread 为 None

# 新增全局变量用于存储实时性能指标，并使用锁保护
realtime_metrics_lock = threading.Lock()
realtime_detections_count = 0
realtime_estimated_fps = 0.0
# 移除实时预处理、推理、后处理时间的全局变量
# realtime_preprocess_time = 0.0
# realtime_inference_time = 0.0
# realtime_postprocess_time = 0.0


# 尝试设置的摄像头分辨率 (宽, 高)
# 根据您的摄像头支持的分辨率进行调整
CAMERA_RESOLUTION = (640, 480) # 例如 640x480 (4:3 比例)

def list_cameras():
    """列出系统中可用的摄像头"""
    available_cameras = []
    # 尝试打开索引 0 到 9 的摄像头
    for i in range(10):
        # 尝试多种后端和方式来检测摄像头
        cap = None
        try:
            # 1. 尝试不指定后端
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                # 检查是否已经添加过同一个索引的摄像头
                if not any(cam['index'] == i for cam in available_cameras):
                    # 尝试获取更详细的摄像头名称（如果可用）
                    backend_name = cap.getBackendName()
                    camera_name = f'摄像头 {i} ({backend_name})' if backend_name else f'摄像头 {i} (Default)'
                    available_cameras.append({'index': i, 'name': camera_name})
                cap.release()
                continue # 找到一个就继续下一个索引

            # 2. 尝试 DSHOW 后端 (Windows)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                if not any(cam['index'] == i for cam in available_cameras):
                     backend_name = cap.getBackendName()
                     camera_name = f'摄像头 {i} ({backend_name})' if backend_name else f'摄像头 {i} (DSHOW)'
                     available_cameras.append({'index': i, 'name': camera_name})
                cap.release()
                continue

            # 3. 尝试 MSMF 后端 (Windows Media Foundation,较新)
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap is not None and cap.isOpened():
                 if not any(cam['index'] == i for cam in available_cameras):
                     backend_name = cap.getBackendName()
                     camera_name = f'摄像头 {i} ({backend_name})' if backend_name else f'摄像头 {i} (MSMF)'
                     available_cameras.append({'index': i, 'name': camera_name})
                 cap.release()
                 continue

        except Exception as e:
            # 忽略检测过程中的错误
            if cap is not None:
                cap.release()
            pass # 继续尝试下一个索引或后端

    logger.info(f"检测到 {len(available_cameras)} 个可用摄像头。")
    return available_cameras


def load_camera_model(model_config, model_name):
    """加载摄像头检测模型"""
    global model_for_camera, camera_model_path, current_model_name

    logger.info(f"尝试加载摄像头模型: {model_name}")
    if model_name not in model_config:
         logger.error(f"无效的模型名称: {model_name}")
         model_for_camera = None
         camera_model_path = None
         current_model_name = None
         return False, f"无效的模型名称: {model_name}"

    model_path_str = model_config.get(model_name) # 获取模型路径字符串
    model_path = Path(model_path_str) # 转换为 Path 对象
    if not model_path.exists(): # 使用 Path 对象检查文件是否存在
        logger.error(f"未找到模型文件: {model_path}，摄像头功能将无法使用。")
        model_for_camera = None
        camera_model_path = None
        current_model_name = None
        return False, f"未找到模型文件: {model_path}"

    # 检查是否已经加载了相同的模型
    if camera_model_path == model_path and model_for_camera is not None: # 比较 Path 对象
         logger.info(f"模型 {model_name} 已加载。")
         current_model_name = model_name
         return True, None


    try:
        # 使用 image_detector 中的 load_model 函数加载模型
        # load_model 函数内部会处理 .pt, .engine 和 .pth 格式
        model_for_camera = load_model(model_path) # 传递 Path 对象
        if model_for_camera is None:
             # load_model 内部已经记录了错误日志
             return False, f"加载模型失败: {model_path_str}"

        camera_model_path = model_path # 记录加载成功的模型路径 (Path 对象)
        current_model_name = model_name # 记录当前使用的模型名称
        logger.info(f"摄像头模型 {model_name} 加载成功。")
        return True, None

    except Exception as e:
        logger.error(f"加载摄像头检测模型失败: {str(e)}", exc_info=True)
        model_for_camera = None
        camera_model_path = None
        current_model_name = None
        return False, f"加载模型失败: {str(e)}"


# 摄像头实时检测处理函数 (运行在单独的线程中)
def process_camera_feed():
    global camera, output_frame, lock, model_for_camera, is_camera_running, current_camera_id
    global realtime_detections_count, realtime_estimated_fps, realtime_metrics_lock # 引入全局变量和锁
    # 移除实时预处理、推理、后处理时间的全局变量
    # global realtime_preprocess_time, realtime_inference_time, realtime_postprocess_time # 引入新的实时指标

    logger.info("摄像头处理线程启动。")

    # 确保模型已加载
    if model_for_camera is None:
         logger.error("摄像头检测模型未加载，处理线程退出。")
         is_camera_running = False
         # 在线程退出前清理 output_frame
         with lock:
             output_frame = None
         return

    # 确保摄像头已打开
    if camera is None or not camera.isOpened():
         logger.error("摄像头未打开，处理线程退出。")
         is_camera_running = False
         # 在线程退出前清理 output_frame
         with lock:
             output_frame = None
         return

    # 用于估算帧率和性能指标
    start_time = time.perf_counter()
    frame_count = 0
    # 获取模型类型
    model_type = getattr(model_for_camera, 'model_type', 'unknown')
    logger.info(f"摄像头处理线程使用模型类型: {model_type}")


    while is_camera_running: # 使用标记控制循环
        ret, frame = camera.read()
        if not ret:
            logger.warning("无法从摄像头读取帧。尝试重新打开摄像头。")
            # 如果无法读取帧，尝试重新打开摄像头
            if camera is not None:
                 camera.release()
                 camera = None

            # 尝试多种方式重新打开摄像头
            opened = False
            try:
                # 1. 尝试不指定后端
                # 在重新打开时使用 current_camera_id
                camera = cv2.VideoCapture(int(current_camera_id))
                if camera is not None and camera.isOpened():
                    logger.info(f"成功重新打开摄像头 {current_camera_id} (Default backend)")
                    # 尝试设置分辨率
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                    logger.info(f"尝试设置摄像头分辨率为 {CAMERA_RESOLUTION}")
                    opened = True
                else:
                    # 2. 尝试 DSHOW 后端 (Windows)
                    # 在重新打开时使用 current_camera_id
                    camera = cv2.VideoCapture(int(current_camera_id), cv2.CAP_DSHOW)
                    if camera is not None and camera.isOpened():
                         logger.info(f"成功重新打开摄像头 {current_camera_id} (DSHOW后端)")
                         # 尝试设置分辨率
                         camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                         logger.info(f"尝试设置摄像头分辨率为 {CAMERA_RESOLUTION}")
                         opened = True
                    else:
                         # 3. 尝试 MSMF 后端 (Windows Media Foundation,较新)
                         # 在重新打开时使用 current_camera_id
                         camera = cv2.VideoCapture(int(current_camera_id), cv2.CAP_MSMF)
                         if camera is not None and camera.isOpened():
                             logger.info(f"成功重新打开摄像头 {current_camera_id} (MSMF后端)")
                             # 尝试设置分辨率
                             camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                             camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                             logger.info(f"尝试设置摄像头分辨率为 {CAMERA_RESOLUTION}")
                             opened = True

            except Exception as e:
                 logger.error(f"重新打开摄像头 {current_camera_id} 时发生错误: {str(e)}")
                 if camera is not None:
                     camera.release()
                     camera = None
                 opened = False


            if not opened:
                 logger.error(f"无法重新打开摄像头 {current_camera_id}.")
                 is_camera_running = False # 停止处理线程
                 break # 退出循环
            else:
                 continue # 成功重新打开后继续处理下一帧


        # 在帧上进行预测（如果模型已加载）
        detections_count = 0
        # 移除对预处理、推理、后处理时间的变量定义
        # preprocess_t = 0.0
        # inference_t = 0.0
        # postprocess_t = 0.0
        annotated_frame = frame # 默认使用原始帧

        try:
            # 根据模型类型调用相应的预测逻辑
            model_type = getattr(model_for_camera, 'model_type', 'unknown')

            if model_type == 'ssd':
                # 使用 SSD 模型预测逻辑
                # predict_ssd_image 内部会处理预测和性能指标
                detections, performance = predict_ssd_image(model_for_camera, frame)
                # 如果 predict_ssd_image 返回 None 或空列表，则不绘制
                if detections:
                    annotated_frame = draw_ssd_detections(frame, detections, class_names=SSD_CLASS_NAMES)
                else:
                    annotated_frame = frame # 没有检测结果时使用原帧

                detections_count = performance.get('detections', 0)
                # 移除对预处理、推理、后处理时间的更新
                # preprocess_t = performance.get('preprocess', 0)
                # inference_t = performance.get('inference', 0)
                # postprocess_t = performance.get('postprocess', 0)


            elif model_type == 'yolo':
                # 使用 YOLO 模型预测逻辑 (ultralytics)
                start_predict = time.perf_counter()
                results = model_for_camera.predict(frame, conf=0.2, verbose=False)
                end_predict = time.perf_counter()
                # predict_duration = (end_predict - start_predict) * 1000 # 移除不使用的变量

                # 提取性能指标 (ultralytics speed)
                speed = results[0].speed if results and len(results) > 0 and hasattr(results[0], 'speed') else {}
                detections_count = len(results[0].boxes) if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None else 0

                # 移除对预处理、推理、后处理时间的更新
                # preprocess_t = speed.get('preprocess', 0)
                # inference_t = speed.get('inference', 0)
                # postprocess_t = speed.get('postprocess', 0)
                # 如果无法获取速度指标，则将总预测时间分配给推理时间
                # if not speed:
                #      inference_t = predict_duration


                # 绘制边界框 (使用 ultralytics plot 方法)
                if results and len(results) > 0:
                     annotated_frame = results[0].plot()
                else:
                     annotated_frame = frame # 没有检测结果时使用原帧

            else:
                logger.warning(f"不支持的模型类型 '{model_type}'，跳过帧处理。")
                # 使用原帧，0 检测结果，0 性能指标
                detections_count = 0
                # 移除对预处理、推理、后处理时间的变量赋值
                # preprocess_t = 0.0
                # inference_t = 0.0
                # postprocess_t = 0.0
                annotated_frame = frame


        except Exception as e:
             logger.warning(f"摄像头帧检测过程中发生错误: {str(e)}")
             annotated_frame = frame # 检测错误时使用原帧
             detections_count = 0 # 错误时检测数为 0
             # 移除对预处理、推理、后处理时间的变量赋值
             # preprocess_t = 0.0
             # inference_t = 0.0
             # postprocess_t = 0.0


        frame_count += 1
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        # 估算帧率和更新实时性能指标 (每秒更新一次或每处理一定帧数更新)
        if elapsed_time >= 1.0: # 至少每 1 秒更新一次
             estimated_fps = frame_count / elapsed_time
             with realtime_metrics_lock:
                 realtime_estimated_fps = estimated_fps
                 # 移除对预处理、推理、后处理时间的更新
                 # 将当前帧的性能指标累加到全局变量 (或者取平均，这里取当前帧的指标)
                 # realtime_preprocess_time = preprocess_t
                 # realtime_inference_time = inference_t
                 # realtime_postprocess_time = postprocess_t

             start_time = current_time # 重置计时器
             frame_count = 0 # 重置帧计数
        else:
             # 在不足一秒的时间间隔内，只更新检测数量
             with realtime_metrics_lock:
                 realtime_detections_count = detections_count
                 # 性能指标不频繁更新，保持上一次的值


        # 将处理后的帧编码为 JPEG 格式
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)

        # 确保成功编码
        if not flag:
            continue

        # 将编码后的图像存储在输出帧中，并使用锁保护
        with lock:
            output_frame = encodedImage.tobytes()

        # 控制帧率（可选，取决于性能需求）
        # time.sleep(0.001) # 可以尝试减少等待时间，取决于处理速度

    logger.info("摄像头处理线程停止。")
    # 在线程退出前清理 output_frame
    with lock:
        output_frame = None


# 生成视频流的函数
def generate_camera_feed():
    global output_frame, lock, is_camera_running
    logger.info("视频流生成器已启动。")
    while is_camera_running: # 使用标志控制循环
        # 使用锁确保线程安全地访问输出帧
        with lock:
            # 检查输出帧是否可用
            if output_frame is None:
                # 如果输出帧为空，可以返回空白图像或等待
                # 在此短暂等待
                time.sleep(0.05)
                continue

            frame = output_frame

        # 使用 yield 返回帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def start_camera(camera_id, model_config, model_name):
    """启动摄像头和处理线程"""
    global camera, camera_thread, is_camera_running, current_camera_id, current_model_name, output_frame
    global realtime_detections_count, realtime_estimated_fps, realtime_metrics_lock # 包含全局变量和锁
    # 移除实时预处理、推理、后处理时间的全局变量
    # global realtime_preprocess_time, realtime_inference_time, realtime_postprocess_time # 包含新的实时指标


    logger.info(f"尝试启动摄像头 {camera_id}，使用模型 {model_name}")

    # 如果摄像头已经在运行，先停止并等待
    if is_camera_running:
        logger.warning("摄像头已在运行，停止现有摄像头。")
        stop_camera()
        # 显式等待停止完成，允许释放线程和资源
        # camera_thread 在 stop_camera 中可能被设置为 None，所以需要重新检查
        if camera_thread is not None and camera_thread.is_alive():
             logger.info("等待现有摄像头处理线程退出...")
             # 设置合理的超时时间以避免无限等待
             camera_thread.join(timeout=5) # 最多等待 5 秒
             if camera_thread.is_alive():
                  logger.warning("现有摄像头处理线程未及时退出。")
             else:
                  logger.info("现有摄像头处理线程已退出。")
             # 无需在此处设置 camera_thread = None，后续会根据需要重新赋值


    # 加载指定模型
    success, msg = load_camera_model(model_config, model_name)
    if not success:
         logger.error(f"模型加载失败，无法启动摄像头: {msg}")
         return False, msg


    # 尝试打开指定摄像头
    opened = False
    try:
        # 1. 尝试不指定后端
        # 使用传入的 camera_id 参数，而不是全局变量 current_camera_id
        camera = cv2.VideoCapture(int(camera_id))
        if camera is not None and camera.isOpened():
            logger.info(f"成功打开摄像头 {camera_id} (Default backend)")
            opened = True
        else:
            # 2. 使用 DSHOW 后端 (Windows)
            # 使用传入的 camera_id 参数
            camera = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
            if camera is not None and camera.isOpened():
                 logger.info(f"成功打开摄像头 {camera_id} (DSHOW backend)")
                 opened = True
            else:
                 # 3. 使用 MSMF 后端 (Windows Media Foundation,较新)
                 # 使用传入的 camera_id 参数
                 camera = cv2.VideoCapture(int(camera_id), cv2.CAP_MSMF)
                 if camera is not None and camera.isOpened():
                     logger.info(f"成功打开摄像头 {camera_id} (MSMF backend)")
                     opened = True

    except Exception as e:
         logger.error(f"启动时打开摄像头 {camera_id} 发生错误: {str(e)}", exc_info=True) # 在日志中添加上下文
         if camera is not None:
             camera.release()
             camera = None
         opened = False


    if not opened:
        is_camera_running = False
        current_camera_id = None
        output_frame = None # 打开摄像头失败时也清理 output_frame
        # 失败时重置实时性能指标
        with realtime_metrics_lock:
            realtime_detections_count = 0
            realtime_estimated_fps = 0.0
            # 移除对实时预处理、推理、后处理时间的重置
            # realtime_preprocess_time = 0.0
            # realtime_inference_time = 0.0
            # realtime_postprocess_time = 0.0
        return False, f"无法打开摄像头 {camera_id}。请检查连接和权限。"

    # 摄像头成功打开，启动处理线程
    is_camera_running = True
    current_camera_id = camera_id # 成功打开后记录当前摄像头ID
    output_frame = None # 启动新线程前清除之前的输出帧
    # 重置实时性能指标
    with realtime_metrics_lock:
        realtime_detections_count = 0
        realtime_estimated_fps = 0.0
        # 移除对实时预处理、推理、后处理时间的重置
        # realtime_preprocess_time = 0.0
        # realtime_inference_time = 0.0
        # realtime_postprocess_time = 0.0

    # 尝试设置摄像头分辨率
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    logger.info(f"尝试设置摄像头分辨率为 {CAMERA_RESOLUTION}")

    # 检查实际设置的分辨率
    actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"实际摄像头分辨率: {actual_width}x{actual_height}")

    # 启动摄像头处理线程（如果尚未启动或已停止）
    # 检查线程是否已启动且仍在运行
    # 只有当 camera_thread 为 None 或已停止时才创建新线程
    if camera_thread is None or not camera_thread.is_alive():
         logger.info("启动新的摄像头处理线程。")
         # 开始一个守护线程，主程序退出时会自动关闭
         camera_thread = threading.Thread(target=process_camera_feed, name="camera_processor", daemon=True)
         camera_thread.start()
         logger.info("摄像头处理线程已启动。")
    else:
         logger.warning("摄像头处理线程意外地已在运行。")
    return True, None # 启动成功

def stop_camera():
    """停止摄像头和处理线程"""
    global camera, is_camera_running, output_frame, camera_thread
    global realtime_detections_count, realtime_estimated_fps, realtime_metrics_lock # 包含全局变量和锁
    # 移除实时预处理、推理、后处理时间的全局变量
    # global realtime_preprocess_time, realtime_inference_time, realtime_postprocess_time # 包含新的实时指标

    logger.info("尝试停止摄像头。")
    if not is_camera_running:
        logger.info("摄像头未运行，无需停止。")
        return

    is_camera_running = False # 设置标志停止摄像头处理线程
    logger.info("设置 is_camera_running = False。")

    # 等待摄像头处理线程结束
    # 检查 camera_thread 是否存在且正在运行
    if camera_thread is not None and camera_thread.is_alive():
        logger.info("等待摄像头处理线程退出...")
        # 显式等待线程结束，设置超时
        camera_thread.join(timeout=5) # 最多等待 5 秒
        if camera_thread.is_alive():
             logger.warning("摄像头处理线程未及时退出。")
        else:
             logger.info("摄像头处理线程已退出。")
        # 停止后将 camera_thread 设置为 None 以便下次创建新线程
        camera_thread = None


    if camera is not None and camera.isOpened():
        logger.info("释放摄像头资源。")
        try:
            camera.release()
            logger.info("摄像头资源已释放。")
        except Exception as e:
            logger.error(f"释放摄像头资源时发生错误: {str(e)}", exc_info=True)
        camera = None
        # output_frame = None # output_frame 现在由 process_camera_feed 线程在退出时清理

    # 停止时重置实时性能指标
    with realtime_metrics_lock:
        realtime_detections_count = 0
        realtime_estimated_fps = 0.0
        # 移除对实时预处理、推理、后处理时间的重置
        # realtime_preprocess_time = 0.0
        # realtime_inference_time = 0.0
        # realtime_postprocess_time = 0.0

    logger.info("摄像头停止流程完成。")

# 新增函数获取实时性能指标
def get_realtime_metrics():
    """获取当前实时性能指标"""
    global realtime_detections_count, realtime_estimated_fps, realtime_metrics_lock
    # 移除实时预处理、推理、后处理时间的全局变量
    # global realtime_preprocess_time, realtime_inference_time, realtime_postprocess_time
    with realtime_metrics_lock:
        # 返回当前存储的指标（仅检测数和帧率）
        return {
            'detections': realtime_detections_count,
            'fps': realtime_estimated_fps
            # 移除对预处理、推理、后处理时间的返回
            # 'preprocess': realtime_preprocess_time,
            # 'inference': realtime_inference_time,
            # 'postprocess': realtime_postprocess_time
        }
