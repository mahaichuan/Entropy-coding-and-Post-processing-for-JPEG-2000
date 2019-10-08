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

import pixelcnn_3D
import pixelcnn_3D_context



def codec(LL, HL_collection, LH_collection, HH_collection, Bits, entropy_resi_num, bit_map, c_HL, c_LH, c_HH):

    with tf.variable_scope('LL'):

        ce_loss = pixelcnn_3D.mask_3D_layer(LL, entropy_resi_num, Bits)

    for j in range(4):

        i = 4-1-j

        c = tf.pow(tf.pow(2,(4-1-i)), 2)

        c = tf.cast(c, dtype=tf.float32)

        with tf.variable_scope('HL'+str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HL_collection[i], c_HL[j], entropy_resi_num, Bits) * c + ce_loss
        with tf.variable_scope('LH'+str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(LH_collection[i], c_LH[j], entropy_resi_num, Bits) * c + ce_loss
        with tf.variable_scope('HH'+str(i)):
            ce_loss = pixelcnn_3D_context.mask_3D_layer(HH_collection[i], c_HH[j], entropy_resi_num, Bits) * c + ce_loss

    return ce_loss * bit_map / 256.