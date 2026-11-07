# Import dependencies

import urllib.request
import numpy as np
import tensorflow as tf

# Download iris dataset

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
with urllib.request.urlopen(dataset_url) as response:
    with open("iris-data.txt", "wb") as out_file:
        out_file.write(response.read())

# Pre-process data

# seed random-generators
rng = np.random.default_rng(seed=42)

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

    rng.shuffle(tmp_list)

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
labels = np.zeros((len(labels), len(tmp_list)), dtype=np.float32)
labels[np.arange(len(labels)), label_idx] = 1.0

# split into train-test set
features_train = np.array(features[:int(train_test_ratio * len(features))], dtype=np.float32)
features_test = np.array(features[int(train_test_ratio * len(features)):], dtype=np.float32)

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

# Weights and biases
layer_1 = {
    'weights':
    tf.Variable(
        tf.random.normal([n_input_layers, n_hidden_layers_1], stddev=0.1)),
    'biases':
    tf.Variable(tf.random.normal([n_hidden_layers_1], stddev=0.1))
}
layer_op = {
    'weights':
    tf.Variable(
        tf.random.normal([n_hidden_layers_1, n_output_layers], stddev=0.1)),
    'biases':
    tf.Variable(tf.random.normal([n_output_layers], stddev=0.1))
}

trainable_variables = [
    layer_1['weights'], layer_1['biases'],
    layer_op['weights'], layer_op['biases']
]


# Model
def model(X):
    h_l1 = tf.matmul(X, layer_1['weights']) + layer_1['biases']
    l1 = tf.nn.tanh(h_l1)

    h_l2 = tf.matmul(l1, layer_op['weights']) + layer_op['biases']
    op = tf.nn.sigmoid(h_l2)
    return op


# Error and Optimizer

# mean-squared error
def compute_error(predictions, labels_batch):
    return tf.reduce_mean(tf.square(predictions - labels_batch))


# gradient-descent-with-momentum-optimizer
optimizer = tf.optimizers.SGD(learning_rate, momentum=momentum)


@tf.function
def train_step(x_sample, y_sample):
    with tf.GradientTape() as tape:
        predictions = model(x_sample)
        loss = compute_error(predictions, y_sample)
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return loss


# Training
print("*********** Train ***********")

# Epoch training
for epoch in range(n_epochs):
    tr_err = []

    for i in range(len(features_train)):
        x_sample = features_train[i:i + 1]
        y_sample = labels_train[i:i + 1]
        error = train_step(x_sample, y_sample)
        tr_err.append(error)

    if epoch % 10 == 0:
        # use test set for validation
        val_preds = model(features_test)
        val_err = compute_error(val_preds, labels_test)
        train_err = sum(tr_err) / len(tr_err)
        print("Epoch:", epoch, " Train-error:", float(train_err),
              " Validation-error:", float(val_err))

print("*********** Test ***********")

test_preds = model(features_test)
test_error = compute_error(test_preds, labels_test)
correct = tf.equal(tf.argmax(test_preds, 1), tf.argmax(labels_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
test_accuracy = accuracy * 100
print('Test-error:', float(test_error), 'Accuracy:', float(test_accuracy))
