import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


#################### Util Functions ####################

def plot_xy_images(images, title, no_i_x, no_i_y):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, 28, 28)
    for i in range(no_i_x):
        for j in range(no_i_y):
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


#################### Preprocess data ####################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
features_train, labels_train, features_test, labels_test = \
    mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#################### Neural Network ####################

# Parameters
n_input_layer = features_train.shape[1]
n_hidden_layer = 500

learning_rate = 1.0

n_epoch = 10
batch_size = 100

# no of images in plot (X, Y co-ordinates)
test_x = 4
test_y = 4


# Sigmoid function
def sigmoid(inp):
    return 1.0 / (1.0 + np.exp(-inp))


# Get sample from input
def get_sample(inp):
    distr = np.random.uniform(size=inp.shape)
    sample = 1.0 * (np.sign(inp - distr) > 0)
    return sample


# Function to train network (CD-k , k=1)
def train(inp, w, b, c):
    # Forward pass
    p_h = sigmoid(np.dot(inp, w) + b)
    s_h = get_sample(p_h)

    # Backward pass
    p_v = sigmoid(np.dot(s_h, w.T) + c)
    s_v = get_sample(p_v)

    p_h1 = sigmoid(np.dot(s_v, w) + b)

    # Error function
    error = np.mean(0.5 * np.square(s_v - inp))

    # Positive phase grad
    p_w_change = np.dot(inp.T, s_h)

    # Negative phase grad
    n_w_change = np.dot(s_v.T, p_h1)

    contr_div = (p_w_change - n_w_change) / inp.shape[0]

    change_w = contr_div
    change_c = np.mean(inp - s_v, 0)
    change_b = np.mean(s_h - p_h1, 0)

    return error, change_w, change_b, change_c


# Regenerate image from input
def regenerate(inp, w, b, c):
    hid = sigmoid(np.dot(inp, w) + b)
    rc = sigmoid(np.dot(hid, w.T) + c)
    return rc


#################### Train Neural Network ####################

# Initialize random  Weights and biases
W = np.random.uniform(0.1, size=(n_input_layer, n_hidden_layer))
B = np.random.uniform(0.1, size=n_hidden_layer)
C = np.random.uniform(0.1, size=n_input_layer)

# Training set
X = features_train
n_batch = X.shape[0] // batch_size
X = np.split(X, n_batch)

# Epoch-training
for epoch in range(n_epoch):
    err = []

    # Batch training
    for b_idx in range(n_batch):
        x = X[b_idx]
        e, grad_w, grad_b, grad_c = train(x, W, B, C)

        # Adjust Weights
        W += learning_rate * grad_w
        B += learning_rate * grad_b
        C += learning_rate * grad_c

        err.append(e)

    print("Epoch: %d, Error: %.8f" % (epoch, sum(err) / len(err)))

#################### Reconstruction ####################

img = []
test_cases = test_x * test_y
for i_no in range(test_cases):
    img.append(regenerate(features_test[i_no], W, B, C))

plot_xy_images(img, "Reconstructed MNIST Data", test_x, test_y)
plot_xy_images(features_test[:test_cases], "Original MNIST Data", test_x, test_y)
plt.show()
