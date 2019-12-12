import numpy as np
import tensorflow as tf
import random
n_batch = None
lr = 0.01
n_step = 10000
n_log = 1000


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.truncated_normal(shape)
    return tf.Variable(init)


np_x = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=np.float32)
np_y = np.array([[0], [1], [0], [1]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[n_batch, 2])
Y = tf.placeholder(tf.float32, shape=[n_batch, 1])

W1 = weight_variable([2, 2])
b1 = bias_variable([2])
tf_out = tf.nn.sigmoid(tf.matmul(X, W1)+b1)

W2 = weight_variable([2, 1])
b2 = bias_variable([1])
tf_out = tf.nn.sigmoid(tf.matmul(tf_out, W2)+b2)
cost = -tf.reduce_mean(Y * tf.log(tf_out) + (1 - Y) * tf.log(1 - tf_out))

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(n_step):

    index = random.randint(0, np_x.shape[0]-1)
    sess.run(train, feed_dict={X: np_x, Y: np_y})

    if step % n_log == 0:

        if step == 0:
            continue

        print(str(step)+'\t\t\t'+str(round(sess.run(cost, feed_dict={X: [np_x[index]], Y: [np_y[index]]}), 3)))

tdata = [[1, 0], [1, 1]]
pred = sess.run(tf_out, feed_dict={X: tdata})
print(tdata)
print(np.round_(pred, 1))
