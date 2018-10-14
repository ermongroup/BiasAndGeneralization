import tensorflow as tf
import numpy as np
import math
import glob
import os, sys, shutil, re


def make_model_path(model_path):
    import subprocess
    if os.path.isdir(model_path):
        subprocess.call(('rm -rf %s' % model_path).split())
    os.makedirs(model_path)


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = tf.nn.relu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


def fc_bn_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = tf.nn.relu(fc)
    return fc


def fc_bn_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = lrelu(fc)
    return fc


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, max_samples=100):
    if max_samples > samples.shape[0]:
        max_samples = samples.shape[0]
    cnt, height, width = int(math.floor(math.sqrt(max_samples))), samples.shape[1], samples.shape[2]
    samples = samples[:cnt*cnt]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


# Input a tensor of shape [batch_size, height, width, color_dim], tiles the tensor for display
# color_dim can be 1 or 3. The tensor should be normalized to lie in [0, 1]
def create_display(tensor, name):
    channels, height, width, color_dim = [tensor.get_shape()[i].value for i in range(4)]
    cnt = int(math.floor(math.sqrt(float(channels))))
    if color_dim >= 3:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 3])
        color_dim = 3
    else:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 1])
        color_dim = 1
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, cnt, cnt, width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * cnt, (width + 2) * cnt, color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


def create_multi_display(tensors, name):
    channels, height, width, color_dim = [tensors[0].get_shape()[i].value for i in range(4)]
    max_columns = 15
    columns = int(math.floor(float(max_columns) / len(tensors)))
    rows = int(math.floor(float(channels) / columns))
    if rows == 0:
        columns = channels
        rows = 1

    for index in range(len(tensors)):
        if color_dim >= 3:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 3])
            color_dim = 3
        else:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 1])
            color_dim = 1
    tensor = tf.stack(tensors)
    tensor = tf.transpose(tensor, [1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [-1, height, width, color_dim])
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, rows, columns * len(tensors), width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * rows, (width + 2) * columns * len(tensors), color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


