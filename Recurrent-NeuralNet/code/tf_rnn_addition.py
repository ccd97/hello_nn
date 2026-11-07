# Import dependencies

import numpy as np
import tensorflow as tf

# Pre-process data

# seed random number generators
rng = np.random.default_rng(seed=42)
tf.random.set_seed(42)

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
    a = rng.integers(0, 128)
    b = rng.integers(0, 128)
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
                              [2, 0, 1]).astype(np.float32)
features_test = np.transpose(features[int(train_test_ratio * len(features)):],
                             [2, 0, 1]).astype(np.float32)

labels_train = np.transpose(labels[:int(train_test_ratio * len(labels))],
                            [1, 0, 2]).astype(np.float32)
labels_test = np.transpose(labels[int(train_test_ratio * len(labels)):],
                           [1, 0, 2]).astype(np.float32)

# Neural Network

# hyper-parameters
n_input_neurons = 2
n_rnn_neurons = 12
n_output_neurons = 1
sequence_len = 8

learning_rate = 0.01

n_epochs = 100

# weights and biases
layer_op = {
    'weight':
    tf.Variable(
        tf.random.normal([n_rnn_neurons, n_output_neurons], stddev=0.1)),
    'bias':
    tf.Variable(tf.random.normal([n_output_neurons], stddev=0.1))
}

# Model
rnn = tf.keras.layers.SimpleRNN(n_rnn_neurons, return_sequences=True)


def model(X):
    # [sequence_len, batch_size, n_input_neurons] -> [batch_size, sequence_len, n_input_neurons]
    x_batch = tf.transpose(X, [1, 0, 2])
    rnn_ops = tf.transpose(rnn(x_batch), [1, 0, 2])

    pred_op = tf.map_fn(
        lambda x:
            tf.nn.sigmoid(tf.matmul(x, layer_op['weight']) + layer_op['bias']),
        rnn_ops)
    return pred_op


# Error and Optimizer

# adam-optimizer
optimizer = tf.optimizers.Adam(learning_rate)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred_op = model(x)
        # mean-squared error
        error = tf.reduce_mean(0.5 * tf.square(pred_op - y))
    trainable_variables = rnn.trainable_variables + [
        layer_op['weight'], layer_op['bias']
    ]
    grads = tape.gradient(error, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return error


# Training
print("########## TRAIN ##########")

# Epoch training
for epoch in range(n_epochs):
    err = train_step(features_train, labels_train)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Error: {float(err):f}")

print("\n########## TEST ##########")

pred_op = model(features_test).numpy()
op = np.transpose((pred_op > 0.5).astype(np.uint8), [1, 0, 2])
op = np.packbits(op)
op = [reverse_bits(x, 8) for x in op]

c = np.transpose(labels_test.astype(np.uint8), [1, 0, 2])
c = np.packbits(c)
c = [reverse_bits(x, 8) for x in c]

a = features_test[:, :, 0]
a = np.transpose(a.astype(np.uint8), [1, 0])
a = np.packbits(a)
a = [reverse_bits(x, 8) for x in a]

b = features_test[:, :, 1]
b = np.transpose(b.astype(np.uint8), [1, 0])
b = np.packbits(b)
b = [reverse_bits(x, 8) for x in b]

success = 0

for i in range(len(op)):

    if c[i] == op[i]:
        success += 1

    if i < disp_test:
        print(f"{a[i]} + {b[i]} = {c[i]} \t --> {str(c[i] == op[i]):>5} ")

print(f"\nSuccess: {success}/{len(op)}, Accuracy = {success / len(op) * 100:f}")
