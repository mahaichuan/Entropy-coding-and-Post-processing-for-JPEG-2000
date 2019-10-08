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

def mask_3D_resiBlock(x, filter_nums):

    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, filter_nums, filter_nums], 0.01)


    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 1, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]

    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, filter_nums, filter_nums])

    mask = tf.cast(mask, dtype=tf.float32)



    w = w * mask



    b = basic_DL_op.bias_variable('bias1', [filter_nums])

    c = basic_DL_op.conv3d(x, w) + b

    c = tf.nn.relu(c)



    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, filter_nums, filter_nums], 0.01)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [filter_nums])

    c = basic_DL_op.conv3d(c, w) + b

    return x + c

def resiBlock_3D(x, filter_nums):

    w = basic_DL_op.weight_variable('conv1_c', [3, 3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias1_c', [filter_nums])

    c = basic_DL_op.conv3d(x, w) + b

    c = tf.nn.relu(c)



    w = basic_DL_op.weight_variable('conv2_c', [3, 3, 3, filter_nums, filter_nums], 0.01)

    b = basic_DL_op.bias_variable('bias2_c', [filter_nums])

    c = basic_DL_op.conv3d(c, w) + b

    return x + c

def mask_3D_layer(x, long_context, entropy_resi_num, out_dim):

    in_dim = 1

    # transform x in to [batch, depth, height, width, in_channel=1]

    x = tf.transpose(x,perm=[0,3,1,2])
    long_context = tf.transpose(long_context,perm=[0,3,1,2])

    label = tf.one_hot(tf.cast(x, dtype = tf.uint8),out_dim)

    x = x / out_dim

    x_shape = x.get_shape().as_list()

    x_n = x_shape[0]
    x_d = x_shape[1]
    x_h = x_shape[2]
    x_w = x_shape[3]

    x = tf.reshape(x,shape=[x_n,x_d,x_h,x_w,1])
    long_context = tf.reshape(long_context,shape=[x_n,x_d,x_h,x_w,1])

    # ceat mask convolution kernels

    w = basic_DL_op.weight_variable('conv1', [3, 3, 3, in_dim, int(2*out_dim)], 0.01)
    w_c = basic_DL_op.weight_variable('conv1_c', [3, 3, 3, in_dim, int(2*out_dim)], 0.01)

    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]

    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, in_dim, int(2*out_dim)])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [int(2*out_dim)])
    b_c = basic_DL_op.bias_variable('bias1_c', [int(2 * out_dim)])

    x = basic_DL_op.conv3d(x, w) + b
    long_context = basic_DL_op.conv3d(long_context, w_c) + b_c


    conv1 = x
    x = x + long_context




    for i in range(entropy_resi_num):
        with tf.variable_scope('resi_block' + str(i)):
            x = mask_3D_resiBlock(x, int(2*out_dim))
            long_context = resiBlock_3D(long_context, int(2*out_dim))
            x = x + long_context

    x = conv1 + x




    w = basic_DL_op.weight_variable('conv2', [3, 3, 3, int(2 * out_dim), out_dim], 0.01)

    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]
             ],
            [[1, 1, 1], [1, 1, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]
             ]
            ]

    mask = tf.reshape(mask, shape=[3, 3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, int(2 * out_dim), out_dim])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [out_dim])

    x = basic_DL_op.conv3d(x, w) + b




    cross_entropy = tf.losses.softmax_cross_entropy (onehot_labels=label, logits=x)

    return cross_entropy
