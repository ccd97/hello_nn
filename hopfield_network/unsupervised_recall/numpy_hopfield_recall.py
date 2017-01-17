import json
import numpy as np

#################### Import Data ####################

with open('data.json') as data_file:
    data = json.load(data_file)

################## Pre Process Data ##################

# train contains perfect data while test contains distorted data
for typ in ['train', 'test']:
    for i in range(len(data[typ])):
        for key, value in data[typ][i].items():
            data[typ][i][key] = [int(x) for x in value.strip().split(',')]

#################### Neural Network ####################

# Function to train the network(in bulk) using Hebbian learning rule
def train(neu, train_data):
    W = np.zeros([neu, neu])
    for data in train_data:
        [(key, value)] = data.items()
        W += np.outer(value, value)
    for i in range(n_neurons):
        W[i][i] = 0
    return W

# Function to restore pattern and test it with it's label (in bulk)
# Use show_output = True to display true and predicted pattern
def test(W, grid, train_data, test_data, show_output=False):
    true_data = {k:v for d in train_data for k,v in d.items()}

    success = 0.0

    for data in test_data:
        [(key, value)] = data.items()
        true_pattern = true_data[key]
        predicted_pattern = retrieve_pattern(W, value)
        if true_pattern == predicted_pattern:
            success += 1.0
        if show_output:
            print_patten(true_pattern, predicted_pattern, grid)

    return(success/len(test_data))

# Function to retrieve individual noisy patterns
def retrieve_pattern(W, test_data, steps=10):
    res = test_data

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(W[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res

# Function to print patters so that they can be compared
def print_patten(arr1, arr2, n):
    print("True\tPredict")
    for i in range(n):
        for j in range(n):
            print('X' if arr1[i * n + j] == 1 else '.', end='')
        print('\t', end='')
        for j in range(n):
            print('X' if arr2[i * n + j] == 1 else '.', end='')
        print()
    print()


# Parameters
grid_size = 5
n_neurons = grid_size ** 2

# Hopfield networks can hold about 0.138*n_neurons for better denoising
# 0.138 * n_neurons = 0.138*25 = 3.45 ~ 3
train_patterns = 3
test_patterns = 16

# Data Partitioning
training_data = data['train'][:train_patterns]
testing_data = data['test'][:test_patterns]

# Train
W = train(n_neurons, training_data)

# Get accuracy via test function
accuracy = test(W, grid_size, training_data, testing_data)

# Print Accuarcy
print("Accuracy of the network is %f" % (accuracy * 100))
