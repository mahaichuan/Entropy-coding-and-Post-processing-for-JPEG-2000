from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from math import log
import numpy as np
from PIL import Image
import scipy.io as sio
import basic_DL_op

def mask_2D_resiBlock(x, filter_nums):

    w = basic_DL_op.weight_variable('conv1', [3, 3, filter_nums, filter_nums], 0.01)


    mask = [[1,1,1],
            [1,1,0],
            [0,0,0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, filter_nums, filter_nums])

    mask = tf.cast(mask, dtype=tf.float32)



    w = w * mask



    b = basic_DL_op.bias_variable('bias1', [filter_nums])

    c = basic_DL_op.conv2d(x, w) + b

    c = tf.nn.relu(c)



    w = basic_DL_op.weight_variable('conv2', [3, 3, filter_nums, filter_nums], 0.01)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [filter_nums])

    c = basic_DL_op.conv2d(c, w) + b

    return x + c

def resiBlock_2D(x, filter_nums):

    w = basic_DL_op.weight_variable('conv1_mag', [3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias1_mag', [filter_nums])

    c = basic_DL_op.conv2d(x, w) + b

    c = tf.nn.relu(c)



    w = basic_DL_op.weight_variable('conv2_mag', [3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias2_mag', [filter_nums])

    c = basic_DL_op.conv2d(c, w) + b

    return x + c

def mask_2D_layer(x, entropy_resi_num, out_dim, mag):

    in_dim = 1 # indicate the heatmap channel = 1

    output_mask = tf.abs(x) # only the the nonzeros sign need to be coded

    x = x + 1 # using 1,2,3 as symbols: -1-->0; 0-->1; 1-->2



    label = tf.one_hot(tf.cast(x, dtype = tf.uint8),out_dim)

    x_shape = label.get_shape().as_list()

    label = tf.reshape(label, shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[4]])
    output_mask = tf.reshape(output_mask, shape=[x_shape[0], x_shape[1], x_shape[2]])



    w = basic_DL_op.weight_variable('conv1', [3, 3, in_dim, int(5*out_dim)], 0.01)
    w_mag = basic_DL_op.weight_variable('conv1_mag', [3, 3, in_dim, int(5*out_dim)], 0.01)

    mask = [[1, 1, 1],
            [1, 0, 0],
            [0, 0, 0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, in_dim, int(5*out_dim)])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [int(5*out_dim)])
    b_mag = basic_DL_op.bias_variable('bias1_mag', [int(5*out_dim)])

    x = basic_DL_op.conv2d(x, w) + b
    mag = basic_DL_op.conv2d(mag, w_mag) + b_mag

    conv1 = x
    x = x + mag


    for i in range(entropy_resi_num):
        with tf.variable_scope('resi_block' + str(i)):
            x = mask_2D_resiBlock(x, int(5*out_dim))
            mag = resiBlock_2D(mag, int(5*out_dim))
            x = x + mag

    x = conv1 + x


    w = basic_DL_op.weight_variable('conv2', [3, 3, int(5 * out_dim), out_dim], 0.01)

    mask = [[1, 1, 1],
            [1, 1, 0],
            [0, 0, 0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, int(5 * out_dim), out_dim])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [out_dim])

    x = basic_DL_op.conv2d(x, w) + b



    x = tf.nn.softmax(x)

    cross_entropy = tf.reduce_sum((label * x), 3)
    cross_entropy = -tf.reduce_mean(tf.log(cross_entropy + 1e-8)*output_mask)

    return cross_entropy
