# Import dependencies

import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Download iris dataset

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
with urllib.request.urlopen(dataset_url) as response:
    with open("iris-data.txt", "wb") as out_file:
        out_file.write(response.read())

# Pre-process data

# seed random-generators
generator = torch.Generator().manual_seed(42)

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

# create dataset and split into train and test sets (80%-20%)
dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32))

train_set, test_set = random_split(dataset, [0.8, 0.2], generator=generator)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

features_test, labels_test = next(iter(test_loader))

# Neural Network

# hyper-parameters
n_input_layers = features_test.shape[1]
n_hidden_layers_1 = 5
n_output_layers = len(tmp_list)

learning_rate = 0.01
momentum = 0.9

n_epochs = 100


# Model
class FeedForwardNN(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.tanh(self.fc1(x))
        op = self.sigmoid(self.fc2(h1))
        return op


model = FeedForwardNN(n_input_layers, n_hidden_layers_1, n_output_layers)

# Error and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training
print("*********** Train ***********")

# Epoch training
for epoch in range(n_epochs):
    tr_err = []
    # set to trainig mode
    model.train()

    for x_sample, y_sample in train_loader:
        # clear gradients
        optimizer.zero_grad()
        # forward pass
        predictions = model(x_sample)
        # calculate loss
        loss = criterion(predictions, y_sample)
        # calculate backpropogation gradient 
        loss.backward()
        # update the model weights
        optimizer.step()
        tr_err.append(loss.item())

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(features_test)
            val_err = criterion(val_preds, labels_test)
        train_err = sum(tr_err) / len(tr_err)
        print("Epoch:", epoch, " Train-error:", float(train_err),
              " Validation-error:", float(val_err.item()))

# Testing
print("*********** Test ***********")

model.eval()
with torch.no_grad():
    test_preds = model(features_test)
    test_error = criterion(test_preds, labels_test)
    correct = torch.eq(torch.argmax(test_preds, 1), torch.argmax(labels_test, 1))
    accuracy = torch.mean(correct.float())
    test_accuracy = accuracy * 100
print('Test-error:', float(test_error.item()), 'Accuracy:', float(test_accuracy.item()))
