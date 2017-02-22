import numpy as np

#################### Pre Process Data ####################

total_test_cases = 100
train_test_ratio = 0.80

np.random.seed(0)

tmp_list = []
features = []
labels = []

for _ in range(total_test_cases):
    a = np.random.randint(0, 128)
    b = np.random.randint(0, 128)
    c = a + b

    features.append([a, b])
    labels.append(c)

features = np.array(features, dtype=np.uint8).reshape(-1, 1)
labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)
features = np.unpackbits(features, axis=1)
labels = np.unpackbits(labels, axis=1)

for i in range(len(labels)):
    tmp_list.append([features[2 * i], features[2 * i + 1]])

features = np.array(tmp_list)

features_train = np.array(features[:int(train_test_ratio * len(features))])
features_test = np.array(features[int(train_test_ratio * len(features)):])

labels_train = labels[:int(train_test_ratio * len(labels))]
labels_test = labels[int(train_test_ratio * len(labels)):]

#################### Neural Network ####################

# Parameters
n_input_layers = 2
n_hidden_layers = 16
n_output_layers = 1
n_sequence = 8

learning_rate = 1

n_epochs = 100

# Activation Functions and their derivative
activation_f = {
    'identity': lambda f_x: f_x,
    'sigmoid': lambda f_x: 1.0 / (1.0 + np.exp(-f_x)),
    'tanh': lambda f_x: np.tanh(f_x),
    'arctan': lambda f_x: np.arctan(f_x),
    'relu': lambda f_x: f_x * (f_x > 0),
    'softplus': lambda f_x: np.log(1 + np.exp(f_x)),
    'sinusoid': lambda f_x: np.sin(f_x),
    'gaussian': lambda f_x: np.exp(-f_x * f_x)
}
activation_f_prime = {
    'identity': lambda f_dx: 1,
    'sigmoid': lambda f_dx: f_dx * (1.0 - f_dx),
    'tanh': lambda f_dx: 1.0 - f_dx ** 2,
    'arctan': lambda f_dx: 1.0 / (1.0 + np.tan(f_dx) ** 2),
    'relu': lambda f_dx: 1.0 * (f_dx > 0),
    'softplus': lambda f_dx: 1.0 - np.exp(-f_dx),
    'sinusoid': lambda f_dx: np.cos(np.arcsin(f_dx)),
    'gaussian': lambda f_dx: -2 * f_dx * np.sqrt(-np.log(f_dx))
}

# Activation Function Parameters
f1 = 'sigmoid'
f2 = 'sigmoid'

act_f1 = activation_f[f1]
act_f2 = activation_f[f2]

act_f1_prime = activation_f_prime[f1]
act_f2_prime = activation_f_prime[f2]

# Initial Random Weights
V = np.random.normal(scale=0.1, size=(n_input_layers, n_hidden_layers))
W = np.random.normal(scale=0.1, size=(n_hidden_layers, n_output_layers))
R = np.random.normal(scale=0.1, size=(n_hidden_layers, n_hidden_layers))

print("############## TRAIN ##############")

# Training-set
X = features_train
Y = labels_train

# Epoch-training
for e in range(n_epochs):

    E = 0

    for i in range(X.shape[0]):

        err = 0

        V_update = np.zeros_like(V)
        W_update = np.zeros_like(W)
        R_update = np.zeros_like(R)

        h_layers = [np.zeros((1, n_hidden_layers))]

        dels = []

        # Forward Pass
        for j in range(n_sequence):

            # Forward Prop
            x = np.array([X[i][0][-j - 1], X[i][1][-j - 1]]).reshape(1, -1)
            y = np.array(Y[i][-j - 1])

            h_inter = np.dot(x, V) + np.dot(h_layers[-1], R)
            h_final = act_f1(h_inter)
            o_inter = np.dot(h_final, W)
            o_final = act_f2(o_inter)

            # Store hidden layer
            h_layers.append(h_final)

            err += (0.5 * np.square(y - o_final))[0][0]

            # Backward Prop
            del_h_o = -np.multiply(y - o_final, act_f2_prime(o_final))

            # Store delta
            dels.append(del_h_o)

            change_h_o = np.dot(h_final.T, del_h_o)
            W_update += change_h_o

        next_del = np.zeros(n_hidden_layers)

        # Backward Propagation through time
        for j in range(n_sequence):
            x = np.array([X[i][0][j], X[i][1][j]]).reshape(1, -1)

            del_h = (np.dot(next_del, R.T) + np.dot(dels[-j - 1], W.T)) * act_f1_prime(h_layers[-j - 1])

            change_h_h = np.dot(h_layers[-j - 2].T, del_h)
            change_i_h = np.dot(x.T, del_h)

            R_update += change_h_h
            V_update += change_i_h

            next_del = del_h

        E += err / n_sequence

        # Adjust Weights
        V -= V_update * learning_rate
        W -= W_update * learning_rate
        R -= R_update * learning_rate

    print("Epoch: %d Error: %f" % (e, E / X.shape[0]))

print("############## TEST ##############")

# Test-set
X = features_test
Y = labels_test

success = 0

# Start Test
for i in range(X.shape[0]):

    a = np.packbits(X[i][0])[0]
    b = np.packbits(X[i][1])[0]

    d = np.packbits(Y[i])[0]

    c = []

    h_layer = np.zeros((1, n_hidden_layers))

    for j in range(n_sequence):
        x = np.array([X[i][0][-j - 1], X[i][1][-j - 1]]).reshape(1, -1)
        y = np.array(Y[i][-j - 1])

        # Forward prop
        h_inter = np.dot(x, V) + np.dot(h_layer, R)
        h_final = act_f1(h_inter)
        o_inter = np.dot(h_final, W)
        o_final = act_f2(o_inter)

        h_layer = h_final

        c.insert(0, (o_final > 0.5).astype(int)[0][0])

    c = np.packbits(c)[0]

    if c == d:
        success += 1

    print("Success: %5s --> %d + %d = %d" % (c == d, a, b, c))

print("Success: %d/%d, Accuracy = %f" % (success, X.shape[0], success / X.shape[0] * 100))
