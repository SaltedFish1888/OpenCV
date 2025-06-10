import cv2
import time
import numpy as np
from datetime import datetime
import os


class FaceDetectionRecorder:
    def __init__(self):
        # 初始化人脸检测器
        self.face_net = self.load_face_detector()

        # 视频录制相关变量
        self.recording = False
        self.video_writer = None
        self.last_face_time = 0
        self.face_disappeared = False

        # 打开默认摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")

        # 获取摄像头参数
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 20

    def load_face_detector(self):
        """加载人脸检测模型"""
        try:
            # 尝试从本地加载模型文件
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"

            if not os.path.exists(proto_path) or not os.path.exists(model_path):
                # 如果本地没有，尝试从OpenCV数据目录加载
                proto_path = os.path.join(cv2.data.haarcascades, 'deploy.prototxt')
                model_path = os.path.join(cv2.data.haarcascades, 'res10_300x300_ssd_iter_140000.caffemodel')

                if not os.path.exists(proto_path) or not os.path.exists(model_path):
                    # 如果仍然没有，尝试从网络下载
                    import urllib.request
                    print("正在下载模型文件...")
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
        """使用DNN检测人脸"""
        if self.face_net is None:
            return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # 置信度阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))

        return faces

    def start_recording(self):
        """开始录像"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, self.fps,
                (self.frame_width, self.frame_height)
            )
            self.recording = True
            print(f"开始录像: {filename}")

    def stop_recording(self):
        """停止录像"""
        if self.recording:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("录像已停止")

    def process_frame(self, frame):
        """处理每一帧"""
        faces = self.detect_faces(frame)
        current_time = time.time()

        # 绘制检测结果
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 检测逻辑
        if len(faces) > 0:
            self.last_face_time = current_time
            self.face_disappeared = False
            self.start_recording()
        else:
            if self.recording and not self.face_disappeared:
                self.face_disappeared = True
                self.stop_recording()
                cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                return False  # 表示人脸已消失

        # 如果正在录像，写入帧
        if self.recording:
            self.video_writer.write(frame)

        # 显示状态
        status = "Recording" if self.recording else "Standby"
        cv2.putText(frame, f"Status: {status}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Face Detection Recorder", frame)
        return True  # 表示继续运行

    def run(self):
        """主循环"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取帧，退出...")
                    break

                should_continue = self.process_frame(frame)
                if not should_continue:
                    break  # 人脸已消失，退出循环

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop_recording()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    recorder = FaceDetectionRecorder()
    recorder.run()