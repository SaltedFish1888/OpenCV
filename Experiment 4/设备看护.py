import cv2
import numpy as np
import time
import os

class DeviceGuardian:
    def __init__(self):
        self.selected_item = None
        self.tracking = False
        self.recording = False
        self.start_time = 0
        self.video_writer = None
        self.alarm_triggered = False
        self.last_print_time = 0
        self.human_detected = False
        self.human_presence_time = 0
        self.last_human_time = 0
        self.buffer_frames = 0
        self.MAX_BUFFER_FRAMES = 15
        self.device_box = None
        self.min_recording_time = 5
        self.face_net = self.load_face_detector()

    def load_face_detector(self):
        """加载人脸检测模型（OpenCV DNN）"""
        try:
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"

            if not os.path.exists(proto_path) or not os.path.exists(model_path):
                proto_path = os.path.join(cv2.data.haarcascades, 'deploy.prototxt')
                model_path = os.path.join(cv2.data.haarcascades, 'res10_300x300_ssd_iter_140000.caffemodel')

                if not os.path.exists(proto_path) or not os.path.exists(model_path):
                    import urllib.request
                    print("正在下载人脸检测模型...")
                    urllib.request.urlretrieve(
                        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                        "deploy.prototxt")
                    urllib.request.urlretrieve(
                        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                        "res10_300x300_ssd_iter_140000.caffemodel")

            return cv2.dnn.readNetFromCaffe(proto_path, model_path)
        except Exception as e:
            print(f"无法加载人脸检测模型: {e}")
            return None

    def detect_faces(self, frame):
        """使用 DNN 检测人脸"""
        if self.face_net is None:
            return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))
        return faces

    def is_face_near_device(self, face_boxes, proximity_threshold=100):
        """判断是否有人脸靠近设备区域"""
        if self.device_box is None:
            return False

        device_center = (self.device_box[0] + self.device_box[2] / 2,
                         self.device_box[1] + self.device_box[3] / 2)

        for (x, y, w, h) in face_boxes:
            face_center = (x + w / 2, y + h / 2)
            distance = np.sqrt((face_center[0] - device_center[0]) ** 2 +
                               (face_center[1] - device_center[1]) ** 2)
            if distance < proximity_threshold + (w + h) / 2:
                return True
        return False

    def select_item(self, event, x, y, flags, param):
        """鼠标事件回调函数：选择设备区域"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_item = {'x': x, 'y': y, 'w': 0, 'h': 0, 'selected': True}
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_item and self.selected_item['selected']:
            self.selected_item['w'] = x - self.selected_item['x']
            self.selected_item['h'] = y - self.selected_item['y']
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_item['selected'] = False
            if self.selected_item['w'] < 0:
                self.selected_item['x'] += self.selected_item['w']
                self.selected_item['w'] = -self.selected_item['w']
            if self.selected_item['h'] < 0:
                self.selected_item['y'] += self.selected_item['h']
                self.selected_item['h'] = -self.selected_item['h']
            self.tracking = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        cv2.namedWindow("Device Guardian")
        cv2.setMouseCallback("Device Guardian", self.select_item)

        print("设备看护系统启动（基于人脸检测）")
        print("提示：请框选监控区域，然后靠近摄像头测试")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break

            # 绘制区域框
            if self.selected_item and self.selected_item['selected']:
                cv2.rectangle(frame,
                              (self.selected_item['x'], self.selected_item['y']),
                              (self.selected_item['x'] + self.selected_item['w'],
                               self.selected_item['y'] + self.selected_item['h']),
                              (0, 255, 0), 2)

            # 确认设备区域
            if self.selected_item and not self.selected_item['selected'] and self.tracking:
                self.device_box = (self.selected_item['x'], self.selected_item['y'],
                                   self.selected_item['w'], self.selected_item['h'])
                self.tracking = False
                print(f"设备区域设置为: {self.device_box}")

            # 检测人脸并判断靠近情况
            if self.device_box is not None:
                faces = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                current_face_detected = self.is_face_near_device(faces)

                # 绘制设备区域
                cv2.rectangle(frame,
                              (self.device_box[0], self.device_box[1]),
                              (self.device_box[0] + self.device_box[2],
                               self.device_box[1] + self.device_box[3]),
                              (0, 255, 0), 2)

                if current_face_detected:
                    self.last_human_time = time.time()
                    self.buffer_frames = 0

                    if not self.human_detected:
                        self.human_detected = True
                        self.human_presence_time = time.time()
                        self.recording = True
                        self.start_time = time.time()
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.video_writer = cv2.VideoWriter(f'recording_{self.start_time}.avi', fourcc, 20.0,
                                                            (frame.shape[1], frame.shape[0]))
                        self.alarm_triggered = True
                        print(f"人脸靠近设备，开始录像: recording_{self.start_time}.avi")

                    elapsed = time.time() - self.human_presence_time
                    cv2.putText(frame, "ALARM! Face near device!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Presence Time: {elapsed:.1f}s", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if time.time() - self.last_print_time > 1:
                        print(f"人脸仍在设备附近，已持续 {elapsed:.1f} 秒")
                        self.last_print_time = time.time()

                else:
                    if self.human_detected:
                        if self.buffer_frames < self.MAX_BUFFER_FRAMES:
                            self.buffer_frames += 1
                        else:
                            if (time.time() - self.start_time) >= self.min_recording_time:
                                self.human_detected = False
                                if self.recording:
                                    self.recording = False
                                    filename = f'recording_{self.start_time}.avi'
                                    self.video_writer.release()
                                    self.video_writer = None
                                    self.alarm_triggered = False
                                    print(f"人脸离开设备区域，停止录像，保存为 {filename}")

            # 写入录像
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)

            # 显示窗口
            cv2.imshow("Device Guardian", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("重置系统")
                self.selected_item = None
                self.device_box = None
                self.tracking = False
                self.recording = False
                self.human_detected = False
                self.buffer_frames = 0
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                self.alarm_triggered = False

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("系统已关闭")

if __name__ == "__main__":
    guardian = DeviceGuardian()
    guardian.run()
