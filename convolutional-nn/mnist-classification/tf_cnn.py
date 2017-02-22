import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###################### Load Data ######################

# Load MNIST Data
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

#################### Neural Network ####################

# Parameters
n_image_width = 28
n_image_height = 28
n_input_pixels = n_image_height * n_image_width
filter_width = 5
filter_height = 5
n_classes = 10  # digits 0-9
n_channels = 1  # black

con_1_features = 16
con_2_features = 32

learning_rate = 0.001

batch_size = 100

# Input/Output Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, n_input_pixels])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

# Layer Weights and biases
conv_lay_1 = {
    'weight': tf.Variable(tf.random_normal([filter_height, filter_width, n_channels, con_1_features], stddev=0.1)),
    'bias': tf.Variable(tf.random_normal([con_1_features], stddev=0.1))}
conv_lay_2 = {
    'weight': tf.Variable(tf.random_normal([filter_height, filter_width, con_1_features, con_2_features], stddev=0.1)),
    'bias': tf.Variable(tf.random_normal([con_2_features], stddev=0.1))}
fc_nn_lay_1 = {'weight': tf.Variable(tf.random_normal([7 * 7 * con_2_features, n_classes], stddev=0.1)),
               'bias': tf.Variable(tf.random_normal([n_classes], stddev=0.1))}

# Model
x_img = tf.reshape(X, [-1, n_image_width, n_image_height, n_channels])  # [batch, height, width, channels]

h_conv_1 = tf.nn.conv2d(x_img, conv_lay_1['weight'], strides=[1, 1, 1, 1], padding='SAME')
h_relu_1 = tf.nn.relu(h_conv_1 + conv_lay_1['bias'])
op_pool_1 = tf.nn.max_pool(h_relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv_2 = tf.nn.conv2d(op_pool_1, conv_lay_2['weight'], strides=[1, 1, 1, 1], padding='SAME')
h_relu_2 = tf.nn.relu(h_conv_2 + conv_lay_2['bias'])
op_pool_2 = tf.nn.max_pool(h_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flat_lay_3 = tf.reshape(op_pool_2, [-1, 7 * 7 * con_2_features])

h_nn_1 = tf.matmul(flat_lay_3, fc_nn_lay_1['weight']) + fc_nn_lay_1['bias']
final_op = tf.nn.sigmoid(h_nn_1)

# Error and Optimizer
error = tf.reduce_mean(0.5 * tf.square(final_op - Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

# Prediction for test
correct_pred = tf.equal(tf.argmax(final_op, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Start Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print("*********** Train ***********")

    train_examples = len(mnist_data.train.images)

    for i in range(train_examples // batch_size):
        train_batch = mnist_data.train.next_batch(batch_size)
        _, err = sess.run([optimizer, error], feed_dict={X: train_batch[0], Y: train_batch[1]})

        if i % 10 == 0:
            test_batch = mnist_data.test.next_batch(batch_size)
            acc = accuracy.eval({X: test_batch[0], Y: test_batch[1]})
            print("Batch: %d Error = %f Accuracy = %f" % (i, err, acc * 100))

    print("*********** Test ***********")

    acc = accuracy.eval({X: mnist_data.validation.images, Y: mnist_data.validation.labels})
    print("Final Accuracy = %f" % (acc * 100))
