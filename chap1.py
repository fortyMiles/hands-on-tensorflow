import tensorflow as tf


# def relu(X):
#     threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
#     return tf.maximum(X, threshold, name='max')
#
#
# X = tf.placeholder(tf.float32, shape=(), name="X")
#
# relus = []
# for relu_index in range(5):
#     with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
#         relus.append(relu(X))
#
# output = tf.add_n(relus, name='output')
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(output.eval(feed_dict={X: 1}))
#

var1 = tf.Variable(0)

new_var1_value = tf.placeholder(dtype=tf.int32, shape=())

var1_assign = tf.assign(var1, new_var1_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(var1.eval())
    var1_assign.eval(feed_dict={new_var1_value: 1})
    print(var1.eval())
