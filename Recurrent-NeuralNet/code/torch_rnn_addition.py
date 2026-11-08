# Import dependencies

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Pre-process data

# seed random number generators
rng = np.random.default_rng(seed=42)
torch.manual_seed(42)

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

# features shape: (1000, 8, 2), labels shape: (1000, 8, 1)
features = np.transpose(np.array(tmp_list), [0, 2, 1])

# create dataset and split into train and test sets
dataset = TensorDataset(
    torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32)
)

train_set, test_set = random_split(
    dataset, [train_test_ratio, 1.0 - train_test_ratio],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

# transpose to [sequence_len, batch_size, in_dim] for RNN
features_train_t = x_train.permute(1, 0, 2)
labels_train_t = y_train.permute(1, 0, 2)
features_test_t = x_test.permute(1, 0, 2)
labels_test_t = y_test.permute(1, 0, 2)

features_test = features_test_t.numpy()
labels_test = labels_test_t.numpy()

# Neural Network

# hyper-parameters
n_input_neurons = 2
n_rnn_neurons = 12
n_output_neurons = 1
sequence_len = 8

learning_rate = 0.01

n_epochs = 100


# Model
class RNNAddition(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.rnn = nn.RNN(in_dim, hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [sequence_len, batch_size, in_dim]
        rnn_ops, _ = self.rnn(x)
        pred_op = self.sigmoid(self.fc(rnn_ops))
        return pred_op


model = RNNAddition(n_input_neurons, n_rnn_neurons, n_output_neurons)

# Error and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_step(x, y):
    # set to trainig mode
    model.train()
    # clear gradients
    optimizer.zero_grad()
    # forward pass
    pred_op = model(x)
    # calculate loss
    loss = 0.5 * criterion(pred_op, y)
    # calculate backpropogation gradient 
    loss.backward()
    # update the model weights
    optimizer.step()
    return loss.item()


# Training
print("########## TRAIN ##########")

# Epoch training
for epoch in range(n_epochs):
    err = train_step(features_train_t, labels_train_t)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Error: {float(err):f}")

print("\n########## TEST ##########")

model.eval()
with torch.no_grad():
    pred_op = model(features_test_t).numpy()
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
