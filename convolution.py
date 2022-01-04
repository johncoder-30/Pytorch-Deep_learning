import torch
import torchvision.datasets
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


class convolution(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(convolution, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(120, outputsize)

    def forward(self, x):
        a = self.pool1(self.conv1(x))
        a = self.pool2(self.conv2(a))
        # print(a.shape)
        a = self.linear1(a.reshape(-1, 16 * 5 * 5))
        a = self.linear2(self.relu1(a))
        return a


model = convolution(28 * 28, 10)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

n = len(train_dataset)
for i in range(10):
    for j, (image, label) in enumerate(train_loader):
        # image = image.reshape(-1,28*28)
        out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(i, loss.item())

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
