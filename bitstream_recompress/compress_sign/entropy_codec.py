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

import pixelcnn_2D
import pixelcnn_2D_context


def codec(LL, HL_collection, LH_collection, HH_collection, Bits, entropy_resi_num, bit_map, mag_LL, mag_HL, mag_LH, mag_HH, c_HL, c_LH, c_HH):

    with tf.variable_scope('LL'):

        ce_loss = pixelcnn_2D.mask_2D_layer(LL, entropy_resi_num, Bits, mag_LL)

    for j in range(4):

        i = 4 - 1 - j

        c = tf.pow(tf.pow(2,(4-1-i)), 2)

        c = tf.cast(c, dtype=tf.float32)

        with tf.variable_scope('HL'+str(i)):
            ce_loss = pixelcnn_2D_context.mask_2D_layer(HL_collection[i], entropy_resi_num, Bits, mag_HL[i], c_HL[j]) * c + ce_loss
        with tf.variable_scope('LH'+str(i)):
            ce_loss = pixelcnn_2D_context.mask_2D_layer(LH_collection[i], entropy_resi_num, Bits, mag_LH[i], c_LH[j]) * c + ce_loss
        with tf.variable_scope('HH'+str(i)):
            ce_loss = pixelcnn_2D_context.mask_2D_layer(HH_collection[i], entropy_resi_num, Bits, mag_HH[i], c_HH[j]) * c + ce_loss

    return ce_loss / 256.