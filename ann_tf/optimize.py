

import os
import sys

import tensorflow as tf
from .util_nn import adam_optimizer


class Optimizer:

    def __init__(self, model, np_x, np_y, **kwargs):

        self._model = model
        self._np_x = np_x
        self._np_y = np_y
        self._optimize = adam_optimizer(**kwargs)

    def train(self, sess, **kwargs):

        n_step = kwargs.pop('n_step', 10000)
        n_log = kwargs.pop('n_log', 1000)

        model = self._model
        loss = model.loss
        minimize = self._optimize.minimize(loss)

        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(n_step):

            sess.run(minimize, feed_dict={model.X: self._np_x, model.Y: self._np_y})

            if step % n_log == 0:

                if step == 0:
                    continue

                print(sess.run(loss, feed_dict={model.X: self._np_x, model.Y: self._np_y}))
