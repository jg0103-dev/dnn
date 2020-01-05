"""

Test Main + OOP + Package

"""

import tensorflow as tf
import numpy as np

from ann_tf.config import Train
from ann_tf.optimize import Optimizer
from ann_tf.model import Model

CF_TRAIN = Train

dic = dict()

dic['n_batch'] = CF_TRAIN.N_BATCH
dic['n_step'] = CF_TRAIN.N_STEP
dic['n_log'] = CF_TRAIN.N_LOG
dic['learning_rate'] = CF_TRAIN.LEARNING_RATE


def run():

    model = Model(**dic)

    np_x = \
        np.array(
            [
                [1, 0],
                [0, 0],
                [0, 1],
                [1, 1]
            ],
            dtype=np.float32)

    np_y = \
        np.array(
            [
                [0],
                [1],
                [0],
                [1]
            ],
            dtype=np.float32)

    optimize = Optimizer(model, np_x, np_y, **dic)
    sess = tf.compat.v1.Session()

    optimize.train(sess, **dic)
    model.predict(sess, np_x)


if __name__ == '__main__':
    run()
