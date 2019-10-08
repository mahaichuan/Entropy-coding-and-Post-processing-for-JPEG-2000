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

def decomposition(LL, Bits, bit_map):

    sign = tf.sign(LL)

    LL = tf.abs(LL)

    x = LL

    for i in range(bit_map):

        x = tf.concat([LL - tf.stop_gradient(tf.floor(LL / Bits) * Bits), x], 3)

        with tf.get_default_graph().gradient_override_map({"Floor": "Identity"}):

            LL = tf.floor(LL / Bits)

    return x[:,:,:,0:-1], sign



def bit_decom(LL, HL_collection, LH_collection, HH_collection, Bits, bit_map):

    all_sign = []

    LL, sign = decomposition(LL, Bits, bit_map)

    all_sign.append(sign)

    for i in range(4):
        HL_collection[i], sign = decomposition(HL_collection[i],Bits,bit_map)
        all_sign.append(sign)
        LH_collection[i], sign = decomposition(LH_collection[i], Bits, bit_map)
        all_sign.append(sign)
        HH_collection[i], sign = decomposition(HH_collection[i], Bits, bit_map)
        all_sign.append(sign)

    return LL, HL_collection, LH_collection, HH_collection, all_sign
