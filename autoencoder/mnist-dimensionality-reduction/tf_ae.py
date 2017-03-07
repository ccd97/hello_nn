import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#################### Preprocess data ####################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
features_train, labels_train, features_test, labels_test = \
    mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#################### Neural Network ####################

# Parameters
n_input_layer = n_output_layer = features_train.shape[1]
n_enc_hidden_1 = n_dec_hidden_3 = 196
n_enc_hidden_2 = n_dec_hidden_2 = 10
n_enc_hidden_3 = n_dec_hidden_1 = 2

learning_rate = 0.01

n_epoch = 20
batch_size = 100

# input/output placeholders
X = tf.placeholder(tf.float32, [None, n_input_layer])
Y = tf.placeholder(tf.float32, [None, n_enc_hidden_2])

# Weights and biases(Encoder)
enc_layer_1 = {'weights': tf.Variable(tf.random_normal([n_input_layer, n_enc_hidden_1], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_enc_hidden_1], stddev=0.1))}
enc_layer_2 = {'weights': tf.Variable(tf.random_normal([n_enc_hidden_1, n_enc_hidden_2], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_enc_hidden_2], stddev=0.1))}
enc_layer_3 = {'weights': tf.Variable(tf.random_normal([n_enc_hidden_2, n_enc_hidden_3], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_enc_hidden_3], stddev=0.1))}

# Weights and biases(Decoder)
dec_layer_1 = {'weights': tf.Variable(tf.random_normal([n_dec_hidden_1, n_dec_hidden_2], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_dec_hidden_2], stddev=0.1))}
dec_layer_2 = {'weights': tf.Variable(tf.random_normal([n_dec_hidden_2, n_dec_hidden_3], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_dec_hidden_3], stddev=0.1))}
dec_layer_3 = {'weights': tf.Variable(tf.random_normal([n_dec_hidden_3, n_output_layer], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_output_layer], stddev=0.1))}

# Model - Consist of two parts

# Encoder Layer 1 (Part A)
h_enc_1 = tf.add(tf.matmul(X, enc_layer_1['weights']), enc_layer_1['biases'])
enc_1 = tf.nn.sigmoid(h_enc_1)

# Encoder Layer 2 (Part A)
h_enc_2 = tf.add(tf.matmul(enc_1, enc_layer_2['weights']), enc_layer_2['biases'])
enc_2 = tf.nn.sigmoid(h_enc_2)

# Encoder Layer 3 (Part B)
h_enc_3 = tf.add(tf.matmul(Y, enc_layer_3['weights']), enc_layer_3['biases'])
enc_3 = tf.nn.sigmoid(h_enc_3)

# Decoder Layer 1 (Part B)
h_dec_1 = tf.add(tf.matmul(enc_3, dec_layer_1['weights']), dec_layer_1['biases'])
dec_1 = tf.nn.sigmoid(h_dec_1)

# Decoder Layer 2 (Part B)
h_dec_2 = tf.add(tf.matmul(dec_1, dec_layer_2['weights']), dec_layer_2['biases'])
dec_2 = tf.nn.sigmoid(h_dec_2)

# Decoder Layer 3 (Part B)
h_dec_3 = tf.add(tf.matmul(dec_2, dec_layer_3['weights']), dec_layer_3['biases'])
dec_3 = tf.nn.sigmoid(h_dec_3)

# Cost and optimizer - Part A - backpropagate labels error
cost1 = tf.reduce_mean(0.5 * tf.square(enc_2 - Y))
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(cost1)

# Cost and optimizer - Part B - backpropagate reconstruction error
cost2 = tf.reduce_mean(0.5 * tf.square(dec_3 - X))
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)

#################### Train Neural Network ####################

# Make Batches
n_batch = features_train.shape[0] // batch_size
n_test_batch = features_test.shape[0] // batch_size
batched_data = np.split(features_train, n_batch)
batched_labels = np.split(labels_train, n_batch)
batched_test_data = np.split(features_test, n_test_batch)

# Start session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Epoch-training
    for epoch in range(n_epoch):
        err1 = []
        err2 = []

        # Batch training
        for b_idx in range(n_batch):
            # Part A
            mid_lay, e1, _1 = sess.run([enc_2, cost1, optimizer1],
                                       feed_dict={X: batched_data[b_idx], Y: batched_labels[b_idx]})
            # Part B
            e2, _2 = sess.run([cost2, optimizer2],
                              feed_dict={X: batched_data[b_idx], Y: mid_lay})

            err1.append(e1)
            err2.append(e2)

        print("Epoch: %d, Error: %.8f" % (epoch, (sum(err1) + sum(err2)) / (len(err1) + len(err2))))

    output_points = []

    for b_idx in range(n_test_batch):
        # enc_3 gives data in 2-dimension
        mid_lay = sess.run(enc_2, feed_dict={X: batched_test_data[b_idx]})
        dim_red = sess.run(enc_3, feed_dict={Y: mid_lay})

        # Apply Logit function to result for better view of data points
        output_points.extend(np.log(dim_red / (1.0 - dim_red)))

    # Plot the scatter points on the graph for visualization
    output_points = np.array(output_points).T
    labels = [np.where(s == 1) for s in labels_test]
    labels = np.array(labels).reshape(labels_test.shape[0], 1)
    plt.scatter(output_points[0], output_points[1], c=labels, cmap="Vega10")
    plt.show()
