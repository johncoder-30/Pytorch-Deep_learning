import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

N = 100
L = 1000
T = 20
x = np.empty((N, L), dtype=np.float32)
x[:] = np.array(range(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1))
y = np.sin(x / T / 1.0, dtype=np.float32)
plt.plot(np.arange(x.shape[1]), y[0, :])
plt.show()

# x = 100, 1000
# y = 100, 1000

class lstmPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(lstmPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)
            outputs.append(out)

        for i in range(future):
            h_t, c_t = self.lstm(out, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        return outputs


# y = 100, 1000
train_input = torch.from_numpy(y[3:, :-1])
train_target = torch.from_numpy(y[3:, 1:])
test_input = torch.from_numpy(y[:3, :-1])
test_target = torch.from_numpy(y[:3, 1:])
model = lstmPredictor()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters(), lr=0.8)

n_steps = 6
for i in range(n_steps):
    print('step', i)

    def closure():
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_target)
        print('loss=', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)

    with torch.no_grad():
        future = 1000
        pred = model(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        y = pred.detach().numpy()
        n = train_input.shape[1]
        plt.subplot(2, 3, i + 1)
        plt.plot(np.arange(n), y[2, :n], 'r')
        plt.plot(np.arange(n, n + future), y[2, n:], 'g')
plt.show()
