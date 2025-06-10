import cv2
import numpy as np
import time
from datetime import datetime

# 全局配置参数
MIN_CONTOUR_AREA = 1000  # 最小轮廓面积阈值，用于过滤小变化
MOTION_TIMEOUT = 5  # 运动停止后继续录像的时间(秒)
SENSITIVITY = 25  # 运动检测灵敏度(1-100)，值越小越敏感

# 全局变量
drawing = False  # 是否正在绘制矩形
roi_start = (0, 0)  # 矩形起始点
roi_end = (0, 0)  # 矩形结束点
roi_selected = False  # 是否已选择ROI
recording = False  # 是否正在录像
out = None  # 录像输出对象
last_motion_time = 0  # 上次检测到运动的时间
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


def draw_rectangle(event, x, y, flags, param):
    """鼠标回调函数，用于绘制矩形区域"""
    global drawing, roi_start, roi_end, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)
            img_copy = param.copy()
            cv2.rectangle(img_copy, roi_start, roi_end, (0, 0, 255), 2)
            cv2.imshow("Security Monitor", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        roi_selected = True
        cv2.rectangle(param, roi_start, roi_end, (0, 0, 255), 2)
        cv2.imshow("Security Monitor", param)


def is_in_rectangle(point, rect_start, rect_end):
    """判断点是否在矩形内"""
    x, y = point
    x1, y1 = rect_start
    x2, y2 = rect_end

    # 确保x1,y1是左上角，x2,y2是右下角
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    return x_min <= x <= x_max and y_min <= y <= y_max


def detect_motion(frame, rect_start=None, rect_end=None):
    """使用背景减除算法检测运动"""
    # 应用背景减除
    fg_mask = background_subtractor.apply(frame)

    # 二值化处理
    _, thresh = cv2.threshold(fg_mask, SENSITIVITY, 255, cv2.THRESH_BINARY)

    # 形态学操作(开运算)去除噪声
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    motion_frame = frame.copy()

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # 始终绘制物体框
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 仅当进入禁入区域时才触发报警
            if rect_start is None or is_in_rectangle(center, rect_start, rect_end):
                motion_detected = True

    return motion_frame, motion_detected


def main():
    global roi_start, roi_end, roi_selected, recording, out, last_motion_time

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 允许摄像头预热
    print("摄像头预热中...")
    for _ in range(30):
        cap.read()
    print("摄像头准备就绪")

    # 获取第一帧用于设置ROI
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        return

    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Security Monitor")
    cv2.setMouseCallback("Security Monitor", draw_rectangle, frame)

    print("请在画面中用鼠标绘制禁入矩形区域，按Enter确认，按q退出")
    print(f"当前灵敏度: {SENSITIVITY} (可在代码中调整SENSITIVITY参数)")

    # 等待用户绘制ROI
    while True:
        cv2.imshow("Security Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter键确认
            if roi_selected:
                break
            else:
                print("请绘制一个有效的矩形区域")

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测运动
        motion_frame, motion_detected = detect_motion(frame, roi_start, roi_end)

        # 绘制禁入区域
        if roi_selected:
            cv2.rectangle(motion_frame, roi_start, roi_end, (0, 0, 255), 2)

        # 如果有运动且不在录像，开始录像
        if motion_detected:
            if not recording:
                # 生成文件名
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                # 获取视频编码和尺寸
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                frame_size = (int(cap.get(3)), int(cap.get(4)))
                out = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
                recording = True
                print(f"[警报] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 检测到入侵禁入区域，开始录像!")

            # 更新最后运动时间
            last_motion_time = time.time()

        # 如果正在录像，写入帧
        if recording:
            out.write(motion_frame)
            # 在画面上显示"Recording"
            cv2.putText(motion_frame, "Recording", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 如果超过设定时间没有检测到运动，停止录像
            if time.time() - last_motion_time > MOTION_TIMEOUT:
                recording = False
                out.release()
                print(f"[信息] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 危险解除，停止录像")

        # 显示画面
        cv2.imshow("Security Monitor", motion_frame)

        # 检查退出键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 清理资源
    if recording:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()