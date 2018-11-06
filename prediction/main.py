import cv2
from keras.models import load_model
import numpy as np
import time
model = load_model('model/yolov2.h5')

with open('model/coco_classes.txt') as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open('model/yolo_anchors.txt') as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

num_classes = len(class_names)
num_anchors = len(anchors)

img = cv2.imread('foto.jpg')

img = cv2.resize(img, (608, 608))
image_data = np.array(img, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

predict_start = time.time()
model_out = model.predict(image_data)
predict_end = time.time()
cv2.imshow('img', img)

print("Prediction time {}".format(predict_end - predict_start))
print("Expected output {}".format(num_anchors * (num_classes + 5)))
print("Output length {}".format(len(model_out)))
print("Model out shape {}".format(model_out.shape))

model_out = np.reshape(model_out, (19, 19, 425))
model_out = np.reshape(model_out, (361, 425))
model_out = np.reshape(model_out, (361, 5, 85))


# równy czas co przy K.run... a może troche przejrzyściej jest to napisane, teraz jeszcze output -> boxes prediction :)