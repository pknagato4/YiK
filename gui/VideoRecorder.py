from  PyQt5 import QtCore
import numpy as np
import cv2

class VideoRecorder(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(0)
        self.cam_x, self.cam_y, self.cam_ch = (0, 0, 0)
        self.timer = QtCore.QBasicTimer()

    def init_video_slot(self, path):
        self.camera.release()
        self.camera = cv2.VideoCapture(path)

    def start_recording(self):
        self.timer.start(0, self)

    def stop_recording(self):
        self.timer.stop()
        img = np.zeros((self.cam_x, self.cam_y, self.cam_ch))
        self.image_data.emit(img)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return
        print(self.camera.get(cv2.CAP_PROP_FPS))
        ret, img = self.camera.read()
        if ret:
            self.cam_x, self.cam_y, self.cam_ch = img.shape
            self.image_data.emit(img)