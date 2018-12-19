from PyQt5 import QtWidgets, QtCore, QtGui
from prediction.Evaluator import Predictor
import time
import numpy as np

class ObjectDetector(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.Pred = None
        self.image = QtGui.QImage()
        self.classes_list = None
        self.data_file = open('runtime.log', 'w')
        self.i = 0

    def init_predictor(self, model_path, coco_path, anchor_path):
        self.Pred = Predictor(model_path, coco_path, anchor_path)

    def set_allowed_classes_model(self, model):
        self.classes_list = model

    def predict(self, image: np.ndarray):
        allowed_classes = {}
        for i in range(self.classes_list.rowCount()):
            item = self.classes_list.item(i)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Unchecked:
                allowed_classes[item.text()] = False
            elif item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                allowed_classes[item.text()] = True

        return self.Pred.get_predicted_image_with_boxes_drawed(image, allowed_classes)

    def image_data_slot(self, image_data):
        start = time.time()
        image_data = self.predict(image_data)
        stop = time.time()
        print('Prediction took: {}'.format(stop - start))
        self.i += 1
        print(self.i)
        if self.i > 100:
            self.data_file.close()
            exit()
        self.data_file.write(str(self.i) + ',' + str(1/(stop-start)) + '\n')
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, channels = image.shape
        bytesPerLine = 3* width

        image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
