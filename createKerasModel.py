import argparse
import os

parser = argparse.ArgumentParser(description="createKerasModel input validation")
parser.add_argument('yolo_cfg_path', help='Path to yolo .cfg file')
parser.add_argument('yolo_weights_path', help='Path to yolo.weights file')
parser.add_argument('keras_h5_path', help='Path where store keras model')

args = parser.parse_args()

## get absolute path
conf_path = os.path.expanduser(args.yolo_cfg_path)
weights_path = os.path.expanduser(args.yolo_weights_path)
model_path = os.path.expanduser(args.keras_h5_path)

## validate_path_extension
assert conf_path.endswith('.cfg'), '{} is not yolo .cfg file'.format(conf_path)
assert weights_path.endswith('.weights'), '{} is not yolo .weights file'.format(weights_path)
assert model_path.endswith('.h5'), '{} is not yolo model file'.format(model_path)