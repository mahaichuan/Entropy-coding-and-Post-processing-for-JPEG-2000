from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import post_block

def graph(x, y):

    # x: the decoded image
    # y: original image

    x = x / 255. - 0.5

    with tf.variable_scope('postprocessing'):

        x = post_block.post(x)

    recon_loss = tf.losses.mean_squared_error(x, y / 255. - 0.5)

    return recon_loss

x = tf.placeholder(tf.float32, [1, 128, 128, 1])

y = tf.placeholder(tf.float32, [1, 128, 128, 1])

recon_loss = graph(x, y)
