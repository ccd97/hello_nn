# Import dependencies

import numpy as np
import random
import tensorflow as tf
import urllib.request

# Download iris dataset

urllib.request.urlretrieve(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "iris-data.txt")

# Pre-process data

# seed random-generators
random.seed(0)
np.random.seed(0)

train_test_ratio = 0.8

tmp_list = []
tmp_set = set()
features = []
labels = []

# text-file to numpy arrays
with open("iris-data.txt") as f:
    for line in f.readlines():
        if not line.isspace():
            tmp_list.append(line)

    random.shuffle(tmp_list)

for line in tmp_list:
    split_line = line.strip().split(',')
    length_line = len(split_line)

    for i in range(length_line - 1):
        split_line[i] = float(split_line[i])

    label = split_line[length_line - 1]
    tmp_set.add(label)

    features.append(split_line[:length_line - 1])
    labels.append(label)

# Scale data
max_val = max([item for i in features for item in i])
min_val = min([item for i in features for item in i])

for i in range(len(features)):
    for j in range(len(features[0])):
        features[i][j] = (features[i][j] - min_val) / (max_val - min_val)

# One-hot encoding
tmp_list = list(tmp_set)
for i in range(len(labels)):
    labels[i] = tmp_list.index(labels[i])

label_idx = np.array(labels)
labels = np.zeros((len(labels), len(tmp_list)))
labels[np.arange(len(labels)), label_idx] = 1

# split into train-test set
features_train = np.array(features[:int(train_test_ratio * len(features))])
features_test = np.array(features[int(train_test_ratio * len(features)):])

labels_train = labels[:int(train_test_ratio * len(labels))]
labels_test = labels[int(train_test_ratio * len(labels)):]

# Neural Network

# hyper-parameters
n_input_layers = len(features_test[0])
n_hidden_layers_1 = 5
n_output_layers = len(tmp_list)

learning_rate = 0.01
momentum = 0.9

n_epochs = 100

# input/output placeholders
X = tf.placeholder(tf.float32, [None, n_input_layers])
Y = tf.placeholder(tf.float32)

# Weights and biases
layer_1 = {
    'weights':
    tf.Variable(
        tf.random_normal([n_input_layers, n_hidden_layers_1], stddev=0.1)),
    'biases':
        tf.Variable(tf.random_normal([n_hidden_layers_1], stddev=0.1))
}
layer_op = {
    'weights':
    tf.Variable(
        tf.random_normal([n_hidden_layers_1, n_output_layers], stddev=0.1)),
    'biases':
        tf.Variable(tf.random_normal([n_output_layers], stddev=0.1))
}

# Model
h_l1 = tf.nn.xw_plus_b(X, layer_1['weights'], layer_1['biases'])
l1 = tf.nn.tanh(h_l1)

h_l2 = tf.nn.xw_plus_b(l1, layer_op['weights'], layer_op['biases'])
op = tf.nn.sigmoid(h_l2)

# Error and Optimizer

# mean-squared error
err = tf.losses.mean_squared_error(predictions=op, labels=Y)

# gradient-descent-with-momentum-optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(err)

# Start Session
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    print("*********** Train ***********")

    # Epoch training
    for epoch in range(n_epochs):

        tr_err = []

        for i in range(len(features_train)):
            _, error = sess.run([optimizer, err], feed_dict={
                    X: features_train[i].reshape(1, -1),
                    Y: labels_train[i]
                })
            tr_err.append(error)

        if epoch % 10 == 0:
            # use test set for validation
            val_err = err.eval({X: features_test, Y: labels_test})
            train_err = sum(tr_err) / len(tr_err)
            print("Epoch:", epoch, " Train-error:", train_err,
                  " Validation-error:", val_err)

    print("*********** Test ***********")

    correct = tf.equal(tf.argmax(op, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    test_error = err.eval({X: features_test, Y: labels_test})
    test_accuracy = accuracy.eval({X: features_test, Y: labels_test}) * 100
    print('Test-error:', test_error, 'Accuracy:', test_accuracy)
