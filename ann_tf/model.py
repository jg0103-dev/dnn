

import os
import sys


import numpy as np
import tensorflow as tf
from .util_nn import weight_variable, bias_variable


class Model:

    def __init__(self, **kwargs):

        n_batch = kwargs.pop('n_batch', 4)

        self.X = tf.compat.v1.placeholder(tf.float32, shape=[n_batch, 2])
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=[n_batch, 1])

        self._pred = self._build()
        self.loss = self._loss()

    def _build(self):

        w1 = weight_variable([2, 10])
        b1 = bias_variable([10])
        out = tf.nn.sigmoid(tf.matmul(self.X, w1)+b1)

        w2 = weight_variable([10, 1])
        b2 = bias_variable([1])
        out = tf.nn.sigmoid(tf.matmul(out, w2)+b2)

        return out

    def _loss(self):

        pred = self._pred

        loss = -tf.reduce_mean(self.Y * tf.math.log(pred) + (1 - self.Y) * tf.math.log(1 - pred))
        return loss

    def predict(self, sess, tdata):

        pred = sess.run(self._pred, feed_dict={self.X: tdata})
        print(tdata)
        print(np.round_(pred, 1))


if __name__ == '__main__':
    print('Done')