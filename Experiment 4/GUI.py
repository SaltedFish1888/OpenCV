import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
from datetime import datetime
import os
from PIL import Image, ImageTk
import threading
import sys
from io import StringIO
import contextlib


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()

    def write(self, string):
        self.buffer.write(string)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()

    def flush(self):
        pass


class SecuritySystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能安防系统")
        self.root.geometry("1200x800")

        # 创建主布局
        self.create_layout()

        # 初始化摄像头
        self.cap = None
        self.is_running = False
        self.current_mode = None

        # 重定向标准输出到文本框
        self.redirect_output()

    def create_layout(self):
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 创建模式选择按钮
        ttk.Button(control_frame, text="人脸检测",
                   command=lambda: self.start_mode("face")).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="光照变化检测",
                   command=lambda: self.start_mode("light")).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="物品看护",
                   command=lambda: self.start_mode("item")).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="设备看护",
                   command=lambda: self.start_mode("device")).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="区域禁入",
                   command=lambda: self.start_mode("area")).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="停止",
                   command=self.stop_camera).pack(fill=tk.X, padx=5, pady=5)

        # 创建右侧显示区域
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建视频显示区域
        self.video_label = ttk.Label(display_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 创建终端输出区域
        terminal_frame = ttk.LabelFrame(display_frame, text="终端输出")
        terminal_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.terminal = tk.Text(terminal_frame, height=10)
        self.terminal.pack(fill=tk.BOTH, expand=True)

    def redirect_output(self):
        # 重定向标准输出到文本框
        self.redirect = RedirectText(self.terminal)
        sys.stdout = self.redirect

    def start_mode(self, mode):
        if self.is_running:
            self.stop_camera()

        self.current_mode = mode
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        self.is_running = True

        # 根据模式启动相应的程序
        if mode == "face":
            from 人形检测 import FaceDetectionRecorder
            self.program = FaceDetectionRecorder()
        elif mode == "light":
            from 光照变化 import main as light_main
            self.program = light_main
        elif mode == "item":
            from 物品看护 import main as item_main
            self.program = item_main
        elif mode == "device":
            from 设备看护 import DeviceGuardian
            self.program = DeviceGuardian()
        elif mode == "area":
            from 区域禁入 import main as area_main
            self.program = area_main

        # 在新线程中运行程序
        self.thread = threading.Thread(target=self.run_program)
        self.thread.daemon = True
        self.thread.start()

    def run_program(self):
        try:
            if self.current_mode in ["face", "device"]:
                self.program.run()
            else:
                self.program()
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("系统已停止")


if __name__ == "__main__":
    root = tk.Tk()
    app = SecuritySystemGUI(root)
    root.mainloop()