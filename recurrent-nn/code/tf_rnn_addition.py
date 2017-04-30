# Import dependencies

import numpy as np
import tensorflow as tf

# Pre-process data

# seed random number generators
np.random.seed(1)
tf.set_random_seed(1)

total_test_cases = 1000
train_test_ratio = 0.8
disp_test = 20

tmp_list = []
features = []
labels = []


# function to reverse bits of a number
def reverse_bits(n, n_bits):
    rev = 0
    for i in range(n_bits):
        if n & (1 << i):
            rev |= 1 << ((n_bits - 1) - i)
    return rev


# generate data (addition)
for i in range(total_test_cases):
    a = np.random.randint(0, 128)
    b = np.random.randint(0, 128)
    c = a + b

    a = reverse_bits(a, 8)
    b = reverse_bits(b, 8)
    c = reverse_bits(c, 8)

    features.append([a, b])
    labels.append(c)

features = np.array(features, dtype=np.uint8).reshape(-1, 1)
labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)
features = np.unpackbits(features, axis=1)
labels = np.unpackbits(labels, axis=1)

labels = np.expand_dims(labels, 2)

for i in range(len(labels)):
    tmp_list.append([features[2 * i], features[2 * i + 1]])

features = np.array(tmp_list)

# split into train-test set and transpose the array
features_train = np.transpose(features[:int(train_test_ratio * len(features))],
                              [2, 0, 1])
features_test = np.transpose(features[int(train_test_ratio * len(features)):],
                             [2, 0, 1])

labels_train = np.transpose(labels[:int(train_test_ratio * len(labels))],
                            [1, 0, 2])
labels_test = np.transpose(labels[int(train_test_ratio * len(labels)):],
                           [1, 0, 2])

# Neural Network

# hyper-parameters
n_input_neurons = 2
n_rnn_neurons = 12
n_output_neurons = 1
sequence_len = 8

learning_rate = 0.01

n_epochs = 100

# input/output placeholders
X = tf.placeholder(tf.float32, [sequence_len, None, n_input_neurons])
Y = tf.placeholder(tf.float32, [sequence_len, None, n_output_neurons])

# weights and biases
layer_op = {
    'weight':
    tf.Variable(
        tf.random_normal([n_rnn_neurons, n_output_neurons], stddev=0.1)),
    'bias':
    tf.Variable(tf.random_normal([n_output_neurons], stddev=0.1))
}

# Model

rnn_cell = tf.contrib.rnn.BasicRNNCell(n_rnn_neurons)
rnn_ops, rnn_states = tf.nn.dynamic_rnn(
    rnn_cell, X, time_major=True, dtype=tf.float32)

pred_op = tf.map_fn(
    lambda x:
        tf.nn.sigmoid(tf.matmul(x, layer_op['weight']) + layer_op['bias']),
        rnn_ops)

# Error and Optimizer

# mean-squared error
error = tf.reduce_mean(0.5 * tf.square(pred_op - Y))

# adam-optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

# Start Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print("########## TRAIN ##########")

    # Epoch training
    for epoch in range(n_epochs):
        _, err = sess.run(
            [optimizer, error], feed_dict={X: features_train,
                                           Y: labels_train})

        if epoch % 10 == 0:
            print("Epoch:", epoch, " Error:", err)

    print("\n########## TEST ##########")

    op = pred_op.eval({X: features_test})
    op = np.transpose((op > 0.5).astype(int), [1, 0, 2])
    op = np.packbits(op)
    op = [reverse_bits(x, 8) for x in op]

    c = np.transpose(labels_test, [1, 0, 2])
    c = np.packbits(c)
    c = [reverse_bits(x, 8) for x in c]

    a = features_test[:, :, 0]
    a = np.transpose(a, [1, 0])
    a = np.packbits(a)
    a = [reverse_bits(x, 8) for x in a]

    b = features_test[:, :, 1]
    b = np.transpose(b, [1, 0])
    b = np.packbits(b)
    b = [reverse_bits(x, 8) for x in b]

    success = 0

    for i in range(len(op)):

        if c == op:
            success += 1

        if i < disp_test:
            print("%d + %d = %d \t --> %5s " % (a[i], b[i], c[i],
                                                c[i] == op[i]))

    print("\nSuccess: %d/%d, Accuracy = %f" % (success, len(op),
                                               success / len(op) * 100))
