import tensorflow as tf


def weight_variable(shape):
    init = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.random.truncated_normal(shape)
    return tf.Variable(init)


def adam_optimizer(**kwargs):
    lr = kwargs.pop('learning_rate', 0.001)
    return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)