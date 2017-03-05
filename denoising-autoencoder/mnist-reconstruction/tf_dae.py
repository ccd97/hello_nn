import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


#################### Util Functions ####################

# Plot mnist images in matplotlib in 3 columns
def plot_xy_images(images, title, no_i_x, no_i_y=3):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, 28, 28)
    for i in range(no_i_x):
        for j in range(no_i_y):
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 0 and i == 1:
                ax.set_title("Distorted")
            elif j == 0 and i == 2:
                ax.set_title("Reconstructed")


# Add noise to input data
def add_noise(data, mean=0, stddev=0.2):
    try:
        noise = np.random.normal(mean, stddev, data.shape)
    except ValueError:
        noise = np.zeros_like(data)

    noisy_data = data + noise
    clipped_noisy_data = np.clip(noisy_data, 0.0, 1.0)

    return clipped_noisy_data


#################### Preprocess data ####################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
features_train, features_test = mnist.train.images, mnist.test.images

#################### Neural Network ####################

# Parameters
n_input_layer = features_train.shape[1]
n_enc_hidden_1 = 392
n_enc_hidden_2 = 196
n_dec_hidden_1 = 196
n_dec_hidden_2 = 392
n_output_layer = features_train.shape[1]

learning_rate = 0.001

n_epoch = 10
batch_size = 100

# no of images in plot (X, Y co-ordinates)
test_disp = 10

# input/output placeholders
X = tf.placeholder(tf.float32, [None, n_input_layer])
Y = tf.placeholder(tf.float32, [None, n_output_layer])

# Weights and biases
enc_layer_1 = {'weights': tf.Variable(tf.random_normal([n_input_layer, n_enc_hidden_1], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_enc_hidden_1], stddev=0.1))}
enc_layer_2 = {'weights': tf.Variable(tf.random_normal([n_enc_hidden_1, n_enc_hidden_2], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_enc_hidden_2], stddev=0.1))}
dec_layer_1 = {'weights': tf.Variable(tf.random_normal([n_dec_hidden_1, n_dec_hidden_2], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_dec_hidden_2], stddev=0.1))}
dec_layer_2 = {'weights': tf.Variable(tf.random_normal([n_dec_hidden_2, n_output_layer], stddev=0.1)),
               'biases': tf.Variable(tf.random_normal([n_output_layer], stddev=0.1))}

# Model
h_enc_1 = tf.add(tf.matmul(X, enc_layer_1['weights']), enc_layer_1['biases'])
enc_1 = tf.nn.sigmoid(h_enc_1)

h_enc_2 = tf.add(tf.matmul(enc_1, enc_layer_2['weights']), enc_layer_2['biases'])
enc_2 = tf.nn.sigmoid(h_enc_2)

h_dec_1 = tf.add(tf.matmul(enc_2, dec_layer_1['weights']), dec_layer_1['biases'])
dec_1 = tf.nn.sigmoid(h_dec_1)

h_dec_2 = tf.add(tf.matmul(dec_1, dec_layer_2['weights']), dec_layer_2['biases'])
dec_2 = tf.nn.sigmoid(h_dec_2)

# Cost and optimizer
cost = tf.reduce_mean(0.5 * tf.square(dec_2 - Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#################### Train Neural Network ####################

# Make Batches
n_batch = features_train.shape[0] // batch_size
batched_data = np.split(features_train, n_batch)

# Start session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Epoch-training
    for epoch in range(n_epoch):
        err = []

        # Batch training
        for b_idx in range(n_batch):
            noisy_data = add_noise(batched_data[b_idx])
            e, _ = sess.run([cost, optimizer], feed_dict={X: batched_data[b_idx], Y: noisy_data})

            err.append(e)

        print("Epoch: %d, Error: %.8f" % (epoch, sum(err) / len(err)))

    # Test the model on test data and try to reconstruct it
    original_imgs = features_test[:test_disp]
    noisy_imgs = add_noise(original_imgs)
    reconstructed_imgs, err = sess.run([dec_2, cost], feed_dict={X: noisy_imgs, Y: original_imgs})
    disp_imgs = []
    for i in range(len(original_imgs)):
        disp_imgs.append(original_imgs[i])
        disp_imgs.append(noisy_imgs[i])
        disp_imgs.append(reconstructed_imgs[i])

    # Plot original, noisy and reconstructed images
    plot_xy_images(disp_imgs, "De-noising Auto-encoder", test_disp)
    print("Test Error: %.8f" % err)
    plt.show()
