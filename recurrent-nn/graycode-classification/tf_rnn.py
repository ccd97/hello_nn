import numpy as np
import tensorflow as tf

#################### Pre Process Data ####################

total_test_cases = 100
train_test_ratio = 0.80

np.random.seed(0)
tf.set_random_seed(0)

tmp_list = []
features = []
labels = []


def graycode(x, mx):
    return int(bin(mx + x ^ int(x / 2))[3:], 2)


for i in range(total_test_cases):
    a = np.random.randint(0, 256)
    b = graycode(a, 256)

    features.append(a)
    labels.append(b)

features = np.array(features, dtype=np.uint8).reshape(-1, 1)
labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)
features = np.unpackbits(features, axis=1)
labels = np.unpackbits(labels, axis=1)

features = np.expand_dims(features, axis=2)
labels = np.expand_dims(labels, axis=2)

features_train = np.transpose(np.array(features[:int(train_test_ratio * len(features))]), [1, 0, 2])
features_test = np.transpose(np.array(features[int(train_test_ratio * len(features)):]), [1, 0, 2])

labels_train = np.transpose(labels[:int(train_test_ratio * len(labels))], [1, 0, 2])
labels_test = np.transpose(labels[int(train_test_ratio * len(labels)):], [1, 0, 2])

#################### Neural Network ####################

# Parameters
n_input_neurons = 1
n_rnn_neurons = 8
n_output_neurons = 1
sequence_len = 8

learning_rate = 0.01

n_epochs = 100

# input/output placeholders
X = tf.placeholder(tf.float32, [sequence_len, None, n_input_neurons])
Y = tf.placeholder(tf.float32, [sequence_len, None, n_output_neurons])

# Weights and biases
layer_op = {
    'weight': tf.Variable(tf.random_normal([n_rnn_neurons, n_output_neurons], stddev=0.1)),
    'bias': tf.Variable(tf.random_normal([n_output_neurons], stddev=0.1))
}

# Model
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_rnn_neurons)
rnn_ops, rnn_states = tf.nn.dynamic_rnn(rnn_cell, X, time_major=True, dtype=tf.float32)

pred_op = tf.map_fn(lambda x: tf.nn.sigmoid(tf.matmul(x, layer_op['weight']) + layer_op['bias']), rnn_ops)

# Cost and optimizer
error = tf.reduce_mean(0.5 * tf.square(pred_op - Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

# Accuracy for test
accuracy = tf.reduce_mean(tf.cast(tf.abs(Y - pred_op) < 0.5, tf.float32))

# Tensor Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print("*********** Train ***********")

    # Epoch training
    for epoch in range(n_epochs):
        _, err = sess.run([optimizer, error], feed_dict={X: features_train, Y: labels_train})
        print("Epoch:", epoch, " Error:", err)

    print("*********** Test ***********")

    acc = accuracy.eval({X: features_test, Y: labels_test})
    print("Accuracy = %f" % (acc * 100))
