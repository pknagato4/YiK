#! /usr/bin/env python

import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Lambda,
                          MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot


def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


parser = argparse.ArgumentParser(
    description='Yet Another Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-flcl',
    '--fully_convolutional',
    help='Model is fully convolutional so set input shape to (None, None, 3). '
    'WARNING: This experimental option does not work properly for YOLO_v2.',
    action='store_true')

class ConfToModelConverter:
    def __init__(self, args):
        self.config_path = ''
        self.weigth_path = ''
        self.output_path = ''
        self.model = self._get_model_(args)
        self.all_layers = None
        self.prev_layer = None
        self.model = None


    def unique_config_sections(self, config_file):
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream


    def _get_model_(self, args):
        output_root = self.validate_paths(args)

        print('Loading weights.')
        weights_file = open(self.weigth_path, 'rb')
        weights_header = np.ndarray(
            shape=(4, ), dtype='int32', buffer=weights_file.read(16))
        print('Weights Header: ', weights_header)

        print('Parsing Darknet config.')
        unique_config_file = self.unique_config_sections(self.config_path)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(unique_config_file)

        print('Creating Keras model.')
        if args.fully_convolutional:
            image_height, image_width = None, None
        else:
            image_height = int(cfg_parser['net_0']['height'])
            image_width = int(cfg_parser['net_0']['width'])
        self.prev_layer = Input(shape=(image_height, image_width, 3))
        self.all_layers = [self.prev_layer]

        weight_decay = float(cfg_parser['net_0']['decay']
                             ) if 'net_0' in cfg_parser.sections() else 5e-4
        count = 0
        for section in cfg_parser.sections():
            print('Parsing section {}'.format(section))
            if section.startswith('convolutional'):
                count = self.add_conv_layer(cfg_parser, count, section, weight_decay, weights_file)

            elif section.startswith('maxpool'):
                self.add_max_pool_layer(cfg_parser, section)

            elif section.startswith('avgpool'):
                self.add_avg_layer(cfg_parser, section)

            elif section.startswith('route'):
                self.add_route_layer(cfg_parser, section)

            elif section.startswith('reorg'):
                self.add_reorg_layer(cfg_parser, section)

            elif section.startswith('region'):
                with open('{}_anchors.txt'.format(output_root), 'w') as f:
                    print(cfg_parser[section]['anchors'], file=f)

            elif (section.startswith('net') or section.startswith('cost') or
                  section.startswith('softmax')):
                pass

            else:
                raise ValueError(
                    'Unsupported section header type: {}'.format(section))

        self.model = Model(inputs=self.all_layers[0], outputs=self.all_layers[-1])
        self.summary(args, count, output_root, weights_file)
        self.model.save('{}'.format(self.output_path))
        print('Saved Keras model to {}'.format(self.output_path))

    def summary(self, args, count, output_root, weights_file):
        print(self.model.summary())
        remaining_weights = len(weights_file.read()) / 4
        weights_file.close()
        print('Read {} of {} from Darknet weights.'.format(count, count +
                                                           remaining_weights))
        if remaining_weights > 0:
            print('Warning: {} unused weights'.format(remaining_weights))
        if args.plot_model:
            plot(self.model, to_file='{}.png'.format(output_root), show_shapes=True)
            print('Saved model plot to {}.png'.format(output_root))

    def add_reorg_layer(self, cfg_parser, section):
        block_size = int(cfg_parser[section]['stride'])
        assert block_size == 2, 'Only reorg with stride 2 supported.'
        self.all_layers.append(
            Lambda(
                space_to_depth_x2,
                output_shape=space_to_depth_x2_output_shape,
                name='space_to_depth_x2')(self.prev_layer))
        self.prev_layer = self.all_layers[-1]

    def add_route_layer(self, cfg_parser, section):
        ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
        layers = [self.all_layers[i] for i in ids]
        if len(layers) > 1:
            print('Concatenating route layers:', layers)
            concatenate_layer = concatenate(layers)
            self.all_layers.append(concatenate_layer)
            self.prev_layer = concatenate_layer
        else:
            skip_layer = layers[0]
            self.all_layers.append(skip_layer)
            self.prev_layer = skip_layer

    def add_avg_layer(self, cfg_parser, section):
        if cfg_parser.items(section) != []:
            raise ValueError('{} with params unsupported.'.format(section))
        self.all_layers.append(GlobalAveragePooling2D()(self.prev_layer))
        self.prev_layer = self.all_layers[-1]

    def add_max_pool_layer(self, cfg_parser, section):
        size = int(cfg_parser[section]['size'])
        stride = int(cfg_parser[section]['stride'])
        self.all_layers.append(
            MaxPooling2D(
                padding='same',
                pool_size=(size, size),
                strides=(stride, stride))(self.prev_layer))
        self.prev_layer = self.all_layers[-1]

    def add_conv_layer(self, cfg_parser, count, section, weight_decay, weights_file):
        filters = int(cfg_parser[section]['filters'])
        size = int(cfg_parser[section]['size'])
        stride = int(cfg_parser[section]['stride'])
        pad = int(cfg_parser[section]['pad'])
        activation = cfg_parser[section]['activation']
        batch_normalize = 'batch_normalize' in cfg_parser[section]
        padding = 'same' if pad == 1 else 'valid'
        prev_layer_shape = K.int_shape(self.prev_layer)
        weights_shape = (size, size, prev_layer_shape[-1], filters)
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print('conv2d', 'bn'
        if batch_normalize else '  ', activation, weights_shape)
        conv_bias = np.ndarray(
            shape=(filters,),
            dtype='float32',
            buffer=weights_file.read(filters * 4))
        count += filters
        if batch_normalize:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weights_file.read(filters * 12))
            count += 3 * filters

            bn_weight_list = [
                bn_weights[0],  # scale gamma
                conv_bias,  # shift beta
                bn_weights[1],  # running mean
                bn_weights[2]  # running var
            ]
        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weights_file.read(weights_size * 4))
        count += weights_size
        conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        conv_weights = [conv_weights] if batch_normalize else [
            conv_weights, conv_bias
        ]
        act_fn = None
        if activation == 'leaky':
            pass
        elif activation != 'linear':
            raise ValueError(
                'Unknown activation function `{}` in section {}'.format(
                    activation, section))
        conv_layer = (Conv2D(
            filters, (size, size),
            strides=(stride, stride),
            kernel_regularizer=l2(weight_decay),
            use_bias=not batch_normalize,
            weights=conv_weights,
            activation=act_fn,
            padding=padding))(self.prev_layer)
        if batch_normalize:
            conv_layer = (BatchNormalization(
                weights=bn_weight_list))(conv_layer)
        self.prev_layer = conv_layer
        if activation == 'linear':
            self.all_layers.append(self.prev_layer)
        elif activation == 'leaky':
            act_layer = LeakyReLU(alpha=0.1)(self.prev_layer)
            self.prev_layer = act_layer
            self.all_layers.append(act_layer)
        return count

    def validate_paths(self, args):
        self.config_path = os.path.expanduser(args.config_path)
        self.weigth_path = os.path.expanduser(args.weights_path)
        assert self.config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
            self.config_path)
        assert self.weigth_path.endswith(
            '.weights'), '{} is not a .weights file'.format(self.weigth_path)
        self.output_path = os.path.expanduser(args.output_path)
        assert self.output_path.endswith(
            '.h5'), 'output path {} is not a .h5 file'.format(self.output_path)
        output_root = os.path.splitext(self.output_path)[0]
        return output_root


if __name__ == '__main__':
    creator = ConfToModelConverter(parser.parse_args())
