from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np
from .keras_yolo import yolo_head, yolo_eval
# import time
import tensorflow as tf


class Predictor:
    def __init__(self, model_path, classes_path, anchors_path):
        self.model = load_model(model_path)
        self.image_size = 224

        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]

        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        self.anchors = anchors
        self.yolo_outputs = yolo_head(self.model.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo_outputs, self.input_image_shape, score_threshold=.39, iou_threshold=.5)
        self.session = K.get_session()
        file_writer = tf.summary.FileWriter(logdir='graph', graph=self.session.graph)

    def predict_boxes(self, session, image_data):
        out_boxes, out_scores, out_classes = session.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [self.image_size, self.image_size],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def draw_boxes(self, img, out_boxes, out_scores, out_classes, allowed_classes):
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if allowed_classes:
                if allowed_classes[predicted_class]:
                    box = out_boxes[i]
                    score = out_scores[i]

                    label = '{} {:.2f}'.format(predicted_class, score)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(self.image_size, np.floor(bottom + 0.5).astype('int32'))
                    right = min(self.image_size, np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))
                    img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
                    cv2.putText(img, label, (left, top + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            else:
                pass
        return img

    def prepare_data(self, img):
        x, y, ch = img.shape
        resized_image = cv2.resize(img, (self.image_size, self.image_size))
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        return resized_image, image_data, (x, y)

    def get_predicted_image_with_boxes_drawed(self, image, allowed_classes=None):
        resized_img, img_data, (x, y) = self.prepare_data(image)
        out_boxes, out_scores, out_classes = self.predict_boxes(self.session, img_data)
        predicted_image = self.draw_boxes(resized_img, out_boxes, out_scores, out_classes, allowed_classes)
        return predicted_image

    def __del__(self):
        self.session.close()

