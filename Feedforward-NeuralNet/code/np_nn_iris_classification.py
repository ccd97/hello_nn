# Import dependencies

import numpy as np
import random
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
n_hidden_layers = 5
n_output_layers = len(tmp_list)

learning_rate = 0.01
momentum = 0.9

n_epoch = 100

# Activation Functions and their derivative
activation_f = {
    'identity': lambda x: x,
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
    'tanh': lambda x: np.tanh(x),
    'arctan': lambda x: np.arctan(x),
    'relu': lambda x: x * (x > 0),
    'softplus': lambda x: np.log(1 + np.exp(x)),
    'sinusoid': lambda x: np.sin(x),
    'gaussian': lambda x: np.exp(-x * x)
}

activation_f_prime = {
    'identity': lambda x: 1,
    'sigmoid': lambda x: x * (1.0 - x),
    'tanh': lambda x: 1 - x**2,
    'arctan': lambda x: 1.0 / (1.0 + np.tan(x)**2),
    'relu': lambda x: 1.0 * (x > 0),
    'softplus': lambda x: 1.0 - np.exp(-x),
    'sinusoid': lambda x: np.cos(np.arcsin(x)),
    'gaussian': lambda x: -2 * x * np.sqrt(-np.log(x))
}

# Activation Function Parameters
f1 = 'tanh'
f2 = 'sigmoid'

act_f1 = activation_f[f1]
act_f2 = activation_f[f2]

act_f1_prime = activation_f_prime[f1]
act_f2_prime = activation_f_prime[f2]


# Training Function
def train(input_features, output_label, i_h_weights, h_o_weights):
    input_features = input_features.reshape(1, -1)

    # forward prop
    h_inter = np.dot(input_features, i_h_weights)
    h_result = act_f1(h_inter)
    o_inter = np.dot(h_result, h_o_weights)
    o_result = act_f2(o_inter)

    error = np.mean(0.5 * np.square(o_result - output_label))

    # back prop
    del_h_o = -np.multiply(output_label - o_result, act_f2_prime(o_result))
    change_h_o = np.dot(h_result.T, del_h_o)
    del_i_h = np.dot(del_h_o, h_o_weights.T) * act_f1_prime(h_result)
    change_i_h = np.dot(input_features.T, del_i_h)

    return error, change_i_h, change_h_o


# Predict Function
def predict(input_features, i_h_weights, h_o_weights):
    # uses just forward prop
    h_inter = np.dot(input_features, i_h_weights)
    h_result = act_f1(h_inter)
    o_inter = np.dot(h_result, h_o_weights)
    o_result = act_f2(o_inter)
    return (o_result >= max(o_result)).astype(int)


# Train Neural Network

print("*********** Train ***********")

# Initial Random Weights
V = np.random.normal(scale=0.1, size=(n_input_layers, n_hidden_layers))
W = np.random.normal(scale=0.1, size=(n_hidden_layers, n_output_layers))

# Training-set
X = features_train
T = labels_train

# Epoch-training
for epoch in range(n_epoch):
    err = []

    for i in range(X.shape[0]):
        loss, grad_V, grad_W = train(X[i], T[i], V, W)

        # Adjust Weights
        V -= learning_rate * grad_V + momentum * grad_V
        W -= learning_rate * grad_W + momentum * grad_W

        err.append(loss)

    if epoch % 10 == 0:
        print("Epoch: %d, Loss: %.8f" % (epoch, sum(err) / len(err)))

# Test Neural Network

print("*********** Test ***********")

success = 0
for i in range(len(features_test)):
    a = predict(features_test[i], V, W)
    b = labels_test[i]
    if np.array_equal(a, b):
        success += 1

print("Total = %d Success = %d Accuracy = %f" %
      (len(features_test), success, success * 100 / len(features_test)))
