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
from math import pow

import bit_decomposition
import entropy_codec
import creat_Long_context

decomposition_step = 4      # num-level of decomposition, in fact, only 4-level decompositions can be used now
Bits = 16                   # range of coefficients {0, 1, ..., Bits - 1}
bit_map = 6                 # number of layers to represent one coefficient
entropy_resi_num = 2
max_value = pow(2,20)

# depart coefficient map into sub-band
def depart_coeff(coeff):

    x_shape = coeff.get_shape().as_list()

    x_h = x_shape[1]
    x_w = x_shape[2]

    HL_collection = []
    LH_collection = []
    HH_collection = []

    for i in range(decomposition_step):

        HL = coeff[:, 0:int(x_h/2), int(x_w/2):x_w, :]
        LH = coeff[:, int(x_h / 2):x_h, 0:int(x_w / 2), :]
        HH = coeff[:, int(x_h / 2):x_h, int(x_w / 2):x_w, :]

        HL_collection.append(HL)
        LH_collection.append(LH)
        HH_collection.append(HH)

        coeff = coeff[:, 0:int(x_h / 2), 0:int(x_w / 2), :]

        x_shape = coeff.get_shape().as_list()

        x_h = x_shape[1]
        x_w = x_shape[2]


    LL = coeff

    return LL, HL_collection, LH_collection, HH_collection

def graph(dwt_recon):

    # creat sub-band

    LL, HL_collection, LH_collection, HH_collection = depart_coeff(dwt_recon)

    # creat long-term context for the magnitude of the current sub-band.

    with tf.variable_scope('context'):

        c_HL, c_LH, c_HH = creat_Long_context.context_all(LL, HL_collection, LH_collection, HH_collection, bit_map, max_value)

    LL, HL_collection, LH_collection, HH_collection, all_sign = bit_decomposition.bit_decom(LL, HL_collection,
                                                                                            LH_collection,
                                                                                            HH_collection, Bits,
                                                                                            bit_map)

    with tf.variable_scope('ce_loss'):

        ce_loss = entropy_codec.codec(LL, HL_collection, LH_collection, HH_collection, Bits, entropy_resi_num, bit_map, c_HL, c_LH, c_HH)

    return ce_loss

x = tf.placeholder(tf.float32, [1, 128, 128, 1])

ce_loss = graph(x)