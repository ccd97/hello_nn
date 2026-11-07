# Import dependencies

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Util functions


# function to plot the images after during testing phase
def plot_images(images, title, no_i_x, no_i_y=2):
    fig = plt.figure(figsize=(5, 15))
    fig.canvas.manager.set_window_title(title)
    images = np.array(images).reshape(-1, 28, 28)
    for i in range(no_i_x):
        for j in range(no_i_y):
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_y * i + (j + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 1 and i == 0:
                ax.set_title("Reconstructed")


# seed random number generators
rng = np.random.default_rng(seed=42)
tf.random.set_seed(42)

# load the mnist dataset
(x_train_raw, _), (x_test_raw, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train_raw.reshape(-1, 28 * 28).astype(np.float32)) / 255.0
x_test = (x_test_raw.reshape(-1, 28 * 28).astype(np.float32)) / 255.0

features_train, features_valid = x_train[:55000], x_train[55000:]
features_test = x_test

# Neural Network Model

# Hyper-parameters
n_input_layer = features_train.shape[1]
n_hidden_layer = 500

learning_rate = 0.05

n_epoch = 10
batch_size = 100

test_disp = 10  # no of images in plot

# Initialize random Weights and biases
W = tf.Variable(
    rng.uniform(0.1, 1.0, size=(n_input_layer, n_hidden_layer)).astype(np.float32))
B = tf.Variable(rng.uniform(0.1, 1.0, size=n_hidden_layer).astype(np.float32))
C = tf.Variable(rng.uniform(0.1, 1.0, size=n_input_layer).astype(np.float32))


# function to get random sample from input
def get_sample(inp):
    distr = tf.random.uniform(shape=tf.shape(inp))
    sample = tf.nn.relu(tf.sign(inp - distr))
    return sample


# Model (Training)
# using contrastive-divergence-k(k = 1)
# x = input
# h(x) = P(h|x) = s_h
# h(x̄) = P(h) = s_h1 (k=1)
# W = W + lr(xᵀh(x) − x̄ᵀh(x̄))
# B = B + lr(h(x) − h(x̄))
# C = C + lr(x − x̄)


@tf.function
def train_step(x):
    # visible -> hidden
    p_h = tf.nn.sigmoid(tf.matmul(x, W) + B)
    s_h = get_sample(p_h)

    # hidden -> visible
    p_v = tf.nn.sigmoid(tf.matmul(s_h, tf.transpose(W)) + C)  # reconstruction
    s_v = get_sample(p_v)

    # visible(1) -> hidden
    p_h1 = tf.nn.sigmoid(tf.matmul(s_v, W) + B)
    s_h1 = get_sample(p_h1)

    # error - just for measuring correctness of reconstructed image
    error = tf.reduce_mean(tf.square(x - p_v))

    # positive and negative phase gradients
    positive_phase = tf.matmul(tf.transpose(x), s_h)
    negative_phase = tf.matmul(tf.transpose(s_v), s_h1)

    contr_div = (positive_phase - negative_phase) / tf.cast(
        tf.shape(x)[0], tf.float32)

    # calculate delta for var
    change_w = contr_div
    change_b = tf.reduce_mean((s_h - s_h1), axis=0)
    change_c = tf.reduce_mean((x - s_v), axis=0)

    # Adjust Weights
    W.assign_add(learning_rate * change_w)
    B.assign_add(learning_rate * change_b)
    C.assign_add(learning_rate * change_c)

    return error


@tf.function
def evaluate(x):
    p_h = tf.nn.sigmoid(tf.matmul(x, W) + B)
    s_h = get_sample(p_h)
    p_v = tf.nn.sigmoid(tf.matmul(s_h, tf.transpose(W)) + C)
    error = tf.reduce_mean(tf.square(x - p_v))
    return error, p_v


# Train Neural Network

# split into batches
n_batch = features_train.shape[0] // batch_size
batched_data = np.split(features_train, n_batch)

disp_imgs = []

print("*********** Train ***********")

# Epoch-training
for epoch in range(n_epoch):
    err = []

    # Batch training
    for b_idx in range(n_batch):
        e = train_step(batched_data[b_idx])
        err.append(float(e))

    val_e, _ = evaluate(features_valid)

    print(
        f"Epoch: {epoch}, Training-error: {sum(err) / len(err):.8f}, Validation-error: {float(val_e):.8f}"
    )

print("*********** Test ***********")

# Test-Reconstruction
test_e, reconstd_image = evaluate(features_test)

print(f"Test-error: {float(test_e):.8f}")

disp_imgs.extend(features_test[:test_disp])
disp_imgs.extend(reconstd_image.numpy()[:test_disp])

# plot output
plot_images(disp_imgs, "Restricted Boltzmann Machine", test_disp)
plt.show()
