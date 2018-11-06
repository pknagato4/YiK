from PyQt5 import QtCore, QtGui, QtWidgets
from .ObjectDetector import ObjectDetector
from .VideoRecorder import VideoRecorder
from .ui import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, paretnt=None):
        super(MainWindow, self).__init__(paretnt)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.camera_source_frame.setVisible(False)
        self.ui.file_source_button.clicked.connect(self.get_file_name)
        self.ui.camera_source_button.clicked.connect(self.show_camera_source_frame)
        self.show()
        self.video_path = ''
        self.model_path = ''
        self.coco_path = ''
        self.anchor_path = ''
        self.object_detection_widget = None
        self.record_video = VideoRecorder()
        self.ui.start_button.clicked.connect(self.start_recording)
        self.ui.img_placeholder.setVisible(False)
        self.ui.model_button.clicked.connect(self.open_model_file)
        self.ui.classes_button.clicked.connect(self.open_classes_file)
        self.ui.anchor_button.clicked.connect(self.open_anchors_file)

    def get_file_name(self):
        file_path = QtWidgets.QFileDialog().getOpenFileName(self, 'Open file', "", "Video format (*.mp4)")
        self.video_path = file_path[0]
        if self.video_path != '':
            self.record_video.init_video_slot(self.video_path)
            self.ui.model_source_frame.setVisible(True)
            self.ui.camera_source_frame.setVisible(False)
            self.ui.source_frame.setVisible(False)

    def show_camera_source_frame(self):
        self.ui.camera_source_frame.setVisible(True)
        self.record_video.init_video_slot(int(self.ui.select_slot.text()))
        self.ui.accept_camera_slot.clicked.connect(self.show_load_model_frame)

    def show_load_model_frame(self):
        self.ui.source_frame.setVisible(False)
        self.ui.camera_source_frame.setVisible(False)
        self.ui.model_source_frame.setVisible(True)

    def open_file(self, label, extension, change_button):
        file_path = QtWidgets.QFileDialog().getOpenFileName(self, 'Open file', "", "{} ({})".format(label, extension))
        file_name = str(file_path[0]).split('/')[-1]
        if file_name != "":
            change_button.setText('{}'.format(file_name))
            change_button.setStyleSheet("background-color: green; color: black")
        else:
            change_button.setText("Failed")
            change_button.setStyleSheet("background-color: red; color: black")
        return file_path

    def open_model_file(self):
        self.model_path = self.open_file('Keras model files', '*.h5', self.ui.model_button)[0]
        self.check_buttons()

    def open_classes_file(self):
        self.coco_path = self.open_file('Coco classes file', '*.txt', self.ui.classes_button)[0]
        self.check_buttons()

    def open_anchors_file(self):
        self.anchor_path = self.open_file('Anchor file', '*.txt', self.ui.anchor_button)[0]
        self.check_buttons()

    def check_buttons(self):
        if self.model_path != '' and self.anchor_path != '' and self.coco_path != '':
            self.ui.start_button.setDisabled(False)
            self.object_detection_widget = ObjectDetector(self.ui.frame)
            self.record_video.image_data.connect(self.object_detection_widget.image_data_slot)
            self.object_detection_widget.init_predictor(self.model_path, self.coco_path, self.anchor_path)
            self.init_list_view()
            self.ui.img_placeholder = self.object_detection_widget
            self.ui.img_placeholder.setVisible(True)
        else:
            self.ui.start_button.setDisabled(True)

    def init_list_view(self):
        list = self.ui.listView
        classes = self.get_classes()
        model = QtGui.QStandardItemModel()
        classes = sorted(classes)
        for obj in classes:
            item = QtGui.QStandardItem(str(obj))
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setData(QtCore.QVariant(QtCore.Qt.Unchecked), QtCore.Qt.CheckStateRole)
            model.appendRow(item)
        list.setModel(model)
        self.object_detection_widget.set_allowed_classes_model(model)


    def start_recording(self):
        if self.ui.start_button.text() == 'Start':
            self.ui.camera_source_frame.setVisible(False)
            self.ui.model_source_frame.setVisible(False)
            self.ui.source_frame.setVisible(False)
            self.record_video.start_recording()
            self.ui.start_button.setText('Stop')
        elif self.ui.start_button.text() == 'Stop':
            self.record_video.stop_recording()
            self.ui.start_button.setText('Start')

    def get_classes(self):
        with open(self.coco_path) as f:
            class_names = f.readlines()
        return [c.strip() for c in class_names]