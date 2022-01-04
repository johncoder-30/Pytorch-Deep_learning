import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
loader = DataLoader(dataset, batch_size=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


model = AutoEncoder(784, 128).to(device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

for i in range(3):
    for (image, label) in loader:
        image=image.to(device=device)
        label=label.to(device=device)
        output = model(image.reshape(-1,28*28))
        loss = criterion(output,image.reshape(-1,784))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss.item())

output=output.cpu().detach().numpy()
image=image.cpu().detach().numpy()

for i in range(6):
    plt.subplot(2,6,i+1)
    plt.imshow(output.reshape(-1,28,28)[i,:,:])
    plt.subplot(2, 6, i+7)
    plt.imshow(image.reshape(-1, 28, 28)[i, :, :])
plt.show()