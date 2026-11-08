# Import dependencies

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# seed random number generators
torch.manual_seed(42)

# Hyper-parameters
learning_rate = 0.001
batch_size = 50

# load the mnist dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(
    root='data', train=False, download=True, transform=transform)

# split into train and validation sets (90%-10%)
train_set, val_set = random_split(train_data, [0.9, 0.1])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Neural Network Model
class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 32)
        x = self.fc(x)
        return x

model = ConvNet()

# Error and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
print("*********** Train ***********")

val_iter = iter(val_loader)

for batch_idx, (data, target) in enumerate(train_loader):
    # set to trainig mode
    model.train()
    # clear gradients
    optimizer.zero_grad()
    # forward pass
    output = model(data)
    # calculate loss
    loss = criterion(output, target)
    # calculate backpropogation gradient 
    loss.backward()
    # update the model weights
    optimizer.step()

    if batch_idx % 100 == 0:
        model.eval()
        with torch.no_grad():
            try:
                val_data, val_target = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_data, val_target = next(val_iter)

            val_output = model(val_data)
            val_loss = criterion(val_output, val_target)
            pred = val_output.argmax(dim=1)
            acc = (pred == val_target).float().mean()
        print("Batch: %d validation-error = %f accuracy = %f" %
              (batch_idx, val_loss.item(), acc.item() * 100))

# Testing
print("*********** Test ***********")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

acc = correct / total
print("Final Accuracy = %f" % (acc * 100))
