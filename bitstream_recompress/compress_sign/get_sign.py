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


def get_sign_fun(x):

    return tf.sign(x)


def gets_sign_all(LL, HL_collection, LH_collection, HH_collection, static_scale):

    sign = get_sign_fun(LL)

    sign_LL = sign
    LL_mag = tf.abs(LL) / static_scale

    sign_HL = []
    sign_LH = []
    sign_HH = []

    HL_mag = []
    LH_mag = []
    HH_mag = []

    for i in range(4):

        sign = get_sign_fun(HL_collection[i])
        sign_HL.append(sign)
        HL_mag.append(tf.abs(HL_collection[i]) / static_scale)

        sign = get_sign_fun(LH_collection[i])
        sign_LH.append(sign)
        LH_mag.append(tf.abs(LH_collection[i]) / static_scale)

        sign = get_sign_fun(HH_collection[i])
        sign_HH.append(sign)
        HH_mag.append(tf.abs(HH_collection[i]) / static_scale)

    return LL_mag, HL_mag, LH_mag, HH_mag, sign_LL, sign_HL, sign_LH, sign_HH
