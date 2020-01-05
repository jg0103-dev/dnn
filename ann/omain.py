"""

Test Main + OOP

"""
import tensorflow as tf
import numpy as np


dic = dict()

dic['n_batch'] = 4
dic['n_step'] = 50000
dic['n_log'] = 1000
dic['learning_rate'] = 0.0001


def weight_variable(shape):
    init = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.random.truncated_normal(shape)
    return tf.Variable(init)


def adam_optimizer(**kwargs):
    lr = kwargs.pop('learning_rate', 0.001)
    return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)


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
        optimizer = adam_optimizer(**kwargs)
        minimize = optimizer.minimize(loss)

        np_x = self._np_x
        np_y = self._np_y

        X = model.X
        Y = model.Y

        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(n_step):

            sess.run(minimize, feed_dict={X: np_x, Y: np_y})

            if step % n_log == 0:

                if step == 0:
                    continue

                print(sess.run(loss, feed_dict={X: np_x, Y: np_y}))


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
        Y = self.Y

        pred = self._pred

        loss = -tf.reduce_mean(Y * tf.math.log(pred) + (1 - Y) * tf.math.log(1 - pred))
        return loss

    def predict(self, sess, tdata):

        pred = sess.run(self._pred, feed_dict={self.X: tdata})
        print(tdata)
        print(np.round_(pred, 1))


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
