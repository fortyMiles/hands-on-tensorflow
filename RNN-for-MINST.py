from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'

n_steps = 28
n_inputs= 28
n_neurons = 150
n_outputs = 10

learning_rate = 1e-3

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = layers.fully_connected(states, n_neurons, activation_fn=None)  # input is states, output is n_neurons

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

mnist = input_data.read_data_sets('/tmp/data/')
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "training accuracy: ", acc_train, 'test accuracy: ', acc_test)

