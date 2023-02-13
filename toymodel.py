from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Networks import models
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST("./data/", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./data/", train=False, transform=transforms.ToTensor())
dataset_size = len(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

model = models.get_model('LeNet')

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

model.to(device)

model.train()

criterion = nn.CrossEntropyLoss()

losses = []

epochs = 10

for epoch in range(epochs):

    train_correct = 0

    for batch in train_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        _, tar = torch.max(output.data, 1)

        train_correct += torch.sum(tar == target.data)

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t correct:{100 * train_correct / dataset_size}")


