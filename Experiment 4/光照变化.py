import cv2
import numpy as np
import time


def detect_light_change(prev_frame, current_frame, threshold=30):
    """
    检测两帧之间的光照变化
    :param prev_frame: 前一帧
    :param current_frame: 当前帧
    :param threshold: 变化阈值
    :return: 是否检测到光照变化
    """
    if prev_frame is None or current_frame is None:
        return False

    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # 计算两帧之间的差异
    frame_diff = cv2.absdiff(prev_gray, current_gray)

    # 计算差异的平均值
    diff_mean = np.mean(frame_diff)

    # 如果平均差异大于阈值，则认为有光照变化
    return diff_mean > threshold


def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 获取摄像头分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 默认值

    # 初始化变量
    prev_frame = None
    recording = False
    recording_start_time = 0
    video_writer = None

    print("开始检测光照变化...")
    print("当检测到明显光照变化时，将自动录像10秒钟")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break

            # 检测光照变化
            if prev_frame is not None and not recording:
                if detect_light_change(prev_frame, frame):
                    print("检测到光照变化，开始录像...")
                    recording = True
                    recording_start_time = time.time()

                    # 初始化视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}.avi"
                    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            # 如果正在录像
            if recording:
                # 写入当前帧
                video_writer.write(frame)

                # 检查是否已经录像10秒
                if time.time() - recording_start_time >= 10:
                    print("录像10秒完成")
                    recording = False
                    video_writer.release()

            # 显示当前帧
            cv2.imshow('Camera', frame)

            # 更新前一帧
            prev_frame = frame.copy()

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        cap.release()
        if recording and video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()