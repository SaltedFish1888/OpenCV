import cv2
import numpy as np
import time
from collections import deque

# 全局变量
selected_item = None
tracking = False
recording = False
start_time = 0
video_writer = None
trajectory = deque(maxlen=50)  # 存储轨迹点
alarm_triggered = False
last_print_time = 0  # 控制报警信息打印频率


# 鼠标回调函数，用于框选物品
def select_item(event, x, y, flags, param):
    global selected_item, tracking

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_item = {'x': x, 'y': y, 'w': 0, 'h': 0, 'selected': True}

    elif event == cv2.EVENT_MOUSEMOVE and selected_item and selected_item['selected']:
        selected_item['w'] = x - selected_item['x']
        selected_item['h'] = y - selected_item['y']

    elif event == cv2.EVENT_LBUTTONUP:
        selected_item['selected'] = False
        # 确保宽度和高度为正数
        if selected_item['w'] < 0:
            selected_item['x'] += selected_item['w']
            selected_item['w'] = -selected_item['w']
        if selected_item['h'] < 0:
            selected_item['y'] += selected_item['h']
            selected_item['h'] = -selected_item['h']
        tracking = True


def main():
    global selected_item, tracking, recording, start_time, video_writer, trajectory, alarm_triggered, last_print_time

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Item Guardian")
    cv2.setMouseCallback("Item Guardian", select_item)

    # 初始化跟踪器
    tracker = cv2.TrackerCSRT_create()

    # 物品原始位置
    original_position = None
    position_threshold = 30  # 移动超过30像素认为物品移动

    print("物品看护系统已启动")
    print("使用说明:")
    print("1. 用鼠标左键框选要看护的物品")
    print("2. 物品移动时会自动开始录像并报警")
    print("3. 按 'r' 键重置系统")
    print("4. 按 'q' 键退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break

        # 显示框选区域
        if selected_item and selected_item['selected']:
            cv2.rectangle(frame,
                          (selected_item['x'], selected_item['y']),
                          (selected_item['x'] + selected_item['w'], selected_item['y'] + selected_item['h']),
                          (0, 255, 0), 2)

        # 如果已经框选物品但未开始跟踪，初始化跟踪器
        if selected_item and not selected_item['selected'] and tracking:
            bbox = (selected_item['x'], selected_item['y'], selected_item['w'], selected_item['h'])
            ok = tracker.init(frame, bbox)
            original_position = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)  # 中心点
            tracking = False  # 已经初始化跟踪器
            print(f"已开始跟踪物品，初始位置: {original_position}")

        # 跟踪物品
        if original_position is not None:
            ok, bbox = tracker.update(frame)

            if ok:
                # 绘制跟踪框
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                # 计算当前中心点
                current_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                # 添加到轨迹
                trajectory.append(current_center)

                # 绘制轨迹
                for i in range(1, len(trajectory)):
                    cv2.line(frame,
                             (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                             (int(trajectory[i][0]), int(trajectory[i][1])),
                             (0, 0, 255), 2)

                # 检查是否移动
                distance = np.sqrt((current_center[0] - original_position[0]) ** 2 +
                                   (current_center[1] - original_position[1]) ** 2)

                if distance > position_threshold:
                    if not recording:
                        # 开始录像
                        recording = True
                        start_time = time.time()
                        # 创建视频写入器
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(f'recording_{start_time}.avi', fourcc, 20.0,
                                                       (frame.shape[1], frame.shape[0]))
                        alarm_triggered = True
                        print(f"警报！物品已移动！开始录像: recording_{start_time}.avi")

                    # 显示报警信息
                    cv2.putText(frame, "ALARM! Item moved!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 显示移动时间
                    elapsed_time = time.time() - start_time
                    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 控制台打印报警信息（每秒打印一次）
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        print(f"物品仍在移动！已移动时间: {elapsed_time:.1f}秒")
                        last_print_time = current_time
                else:
                    if recording:
                        # 停止录像
                        recording = False
                        video_filename = f'recording_{start_time}.avi'
                        video_writer.release()
                        video_writer = None
                        alarm_triggered = False
                        print(f"物品已静止，停止录像。视频已保存为: {video_filename}")
            else:
                # 跟踪失败
                cv2.putText(frame, "Tracking failure!", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                print("警告：物品跟踪失败！")

        # 如果正在录像，写入帧
        if recording and video_writer is not None:
            video_writer.write(frame)

        # 显示帧
        cv2.imshow("Item Guardian", frame)

        # 退出条件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # 重置
            selected_item = None
            tracking = False
            recording = False
            original_position = None
            trajectory.clear()
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            if alarm_triggered:
                alarm_triggered = False
            print("系统已重置，请重新框选物品")

    # 清理
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("物品看护系统已关闭")


if __name__ == "__main__":
    main()