# Import dependencies

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Pre-process data

# seed random number generators
rng = np.random.default_rng(seed=42)
tf.random.set_seed(42)

# Paramaters
seq_len = 50  # Sequence Length
train_test_ratio = 0.7

time_series = []  # float values in ppm
time_stamps = []  # string corresponding to year-month

# read from data from csv file
csv_path = 'datasets/co2-ppm-mauna-loa-19651980.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), '..', csv_path)

with open(csv_path) as f:
    next(f)  # skip header
    for line in f:
        try:
            line_parts = line.strip().split(',')
            time_series.append([float(line_parts[1])])
            time_stamps.append(line_parts[0].strip('"'))
        except Exception:
            break

time_series = np.array(time_series, dtype=np.float32)

# Scale data
ts_min = np.min(time_series)
ts_max = np.max(time_series)

time_series = (time_series - ts_min) / (ts_max - ts_min)

# Split data into train and test
train_time_series = time_series[:int(len(time_series) * train_test_ratio)]
test_time_series = time_series[int(len(time_series) * train_test_ratio) - 1:]


# Creates sequences from data
def create_dataset(data, len_seq):
    features = []
    labels = []
    for i in range(len(data) - len_seq):
        features.append(data[i:i + len_seq])
        labels.append(data[i + len_seq])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


trainX, trainY = create_dataset(train_time_series, seq_len)
testX, testY = create_dataset(
    np.concatenate((trainX[-1], test_time_series)), seq_len)

# Neural Network

# hyper-parameters
n_rnn_neurons = 100
n_input_neurons = 1
n_output_neurons = 1

learn_rate = 0.006

n_epoch = 1000

# Weights and biases for final fully connected layer
layer_op = {
    'weight':
    tf.Variable(
        tf.random.normal([n_rnn_neurons, n_output_neurons], stddev=1.0)),
    'bias':
    tf.Variable(tf.random.normal([n_output_neurons], stddev=1.0))
}

# Model

# lstm + dropout layer
lstm = tf.keras.layers.LSTM(n_rnn_neurons)
dropout = tf.keras.layers.Dropout(0.25)


def model(X, training=False):
    lstm_op = lstm(X)
    if training:
        lstm_op = dropout(lstm_op, training=True)
    final_op = tf.nn.sigmoid(
        tf.matmul(lstm_op, layer_op['weight']) + layer_op['bias'])
    return final_op


# Error and Optimizer

# adam-optimizer
optimizer = tf.optimizers.Adam(learn_rate)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        final_op = model(x, training=True)
        # mean-squared error
        error = tf.reduce_mean(0.5 * tf.square(final_op - y))
    trainable_variables = lstm.trainable_variables + [
        layer_op['weight'], layer_op['bias']
    ]
    grads = tape.gradient(error, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return error


# Training
print("*********** Train ***********")

for epoch in range(n_epoch):
    err = train_step(trainX, trainY)

    if epoch % 100 == 0:
        print(f"Epoch : {epoch} Error = {float(err):f}")

print("\n*********** Test ***********")

resultt = model(testX, training=False)
err = tf.reduce_mean(0.5 * tf.square(resultt - testY))
print(f"Testing Error : {float(err):f}")
resultt = resultt.numpy()

# Predict future values with continuous data
inp = trainX[-1].flatten().tolist()
resultp = []

for i in range(len(test_time_series)):
    x_in = np.reshape(inp, [1, -1, 1]).astype(np.float32)
    op = model(x_in, training=False).numpy()
    inp.append(op[0][0])
    resultp.append(op[0][0])
    del inp[0]

# Plot test and prediction output

plt.figure(figsize=(12, 8))

plt.plot(
    train_time_series * (ts_max - ts_min) + ts_min,
    'b',
    label='training data')
plt.plot(
    np.arange(len(train_time_series) - 1, len(time_series)),
    test_time_series * (ts_max - ts_min) + ts_min,
    'c',
    label='expected data')
plt.plot(
    np.arange(len(train_time_series) - 1, len(time_series)),
    resultt * (ts_max - ts_min) + ts_min,
    'm',
    label='test output')
plt.plot(
    np.arange(len(train_time_series) - 1, len(time_series)),
    np.array(resultp) * (ts_max - ts_min) + ts_min,
    'r',
    label='continous prediction')

plt.xticks(
    np.arange(0, len(time_series), 12),
    time_stamps[::12],
    rotation=70,
    fontsize=7)
plt.xlabel('Month')
plt.ylabel('CO2 (ppm)')
plt.legend(loc='upper left')
plt.show()
