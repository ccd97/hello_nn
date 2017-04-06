import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#################### Pre Process Data ####################

# Paramaters
seq_len = 25  # Sequence Length
train_test_ratio = 0.7

# seed rng
np.random.seed(1)
tf.set_random_seed(1)

time_series = []  # float values in ppm
time_stamps = []  # string corresponding to year-month

# read from data from csv file
with open('datasets/co2-ppm-mauna-loa-19651980.csv') as f:
    skipped_line = False
    for line in f.readlines():
        if not skipped_line:
            skipped_line = True
            continue
        else:
            try:
                line = line.strip().split(',')
                time_series.append([float(line[1])])
                time_stamps.append(line[0].strip('"'))
            except Exception as e:
                break

ts_min = np.min(time_series)
ts_max = np.max(time_series)

# Scale data
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
    return features, labels


# Get features and labels ready
trainX, trainY = create_dataset(train_time_series, seq_len)
testX, testY = create_dataset(
    np.concatenate((trainX[-1], test_time_series)), seq_len)

#################### Neural Network ####################

# Hyper-parameters
n_rnn_neurons = 50
n_input_neurons = 1
n_output_neurons = 1

learn_rate = 0.01

n_epoch = 1000

# Tensorflow placeholders
X = tf.placeholder(tf.float32, [None, seq_len, n_input_neurons])
Y = tf.placeholder(tf.float32, [None, n_output_neurons])

# Weights and biases for final fully connected layer
layer_op = {
    'weight':
    tf.Variable(tf.random_normal([n_rnn_neurons, n_output_neurons], stddev=1)),
    'bias':
    tf.Variable(tf.random_normal([n_output_neurons], stddev=1))
}

# Tensorflow model
cell = tf.contrib.rnn.BasicLSTMCell(n_rnn_neurons)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.7)
lstm_op, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Just connect last output of hidden layer to fully connected layer
op = tf.map_fn(lambda x: x[-1], lstm_op)
final_op = tf.nn.sigmoid(tf.matmul(op, layer_op['weight']) + layer_op['bias'])

# Error and optimizer
error = tf.reduce_mean(0.5 * tf.square(final_op - Y))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(error)

# Tensor Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print("*********** Train ***********")

    for epoch in range(n_epoch):
        _, err = sess.run([optimizer, error], feed_dict={X: trainX, Y: trainY})

        if epoch % 10 == 0:
            print("Epoch : %d Error = %f" % (epoch, err))

    print("*********** Test ***********")

    err, resultt = sess.run([error, final_op], feed_dict={X: testX, Y: testY})
    print("Testing Error : %f" % err)

    # give single starting state and use the predicted output for next input
    print("********** Predict **********")

    inp = trainX[-1].flatten().tolist()
    resultp = []

    for i in range(len(test_time_series)):
        op = sess.run(final_op, feed_dict={X: np.reshape(inp, [1, -1, 1])})
        inp.append(op[0][0])
        resultp.append(op[0][0])
        del inp[0]

    print("Predictions in Graph")

    # Plot graph with matplotlib
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
