"""
The landing algorithm to train a toy neural network on a distilation task
=========================================================================
"""
from time import time

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import geoopt
from geoopt.optim import RiemannianSGD

from landing import LandingSGD


torch.manual_seed(1)


p = 100
n_layers = 10


class Network(nn.Module):
    def __init__(self, n_layers=10):
        super().__init__()
        self.weights = geoopt.ManifoldParameter(
            torch.randn(n_layers, p, p),
            manifold=geoopt.Stiefel(canonical=False),
        )
        with torch.no_grad():
            self.weights.proj_()
        self.biases = torch.nn.Parameter(torch.randn(n_layers, p))
        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers):
            x = torch.tanh(x.mm(self.weights[i].t()) + self.biases[i])
        return x


teacher = Network(n_layers)
# deactivate the parameters in teacher
for param in teacher.parameters():
    param.requires_grad = False


init_weights = torch.randn(n_layers, p, p)
init_biases = torch.randn(n_layers, p)

method_names = ["Landing", "Retraction"]
methods = [LandingSGD, RiemannianSGD]
n_epochs = 100
learning_rate = 0.1
momentum = 0.9
batch_size = 10
batch_per_epoch = 10
test_size = 1000


plt.figure()
for method_name, method in zip(method_names, methods):
    student_network = Network()
    with torch.no_grad():
        student_network.weights.data = init_weights.clone()
        student_network.weights.proj_
        student_network.biases.data = init_biases.clone()
    optimizer_ortho = method(
        (student_network.weights,), lr=learning_rate, momentum=momentum
    )
    optimizer_bias = optim.SGD(
        (student_network.biases,), lr=learning_rate, momentum=momentum
    )
    test_losses = []
    time_epochs = []
    for epoch in range(n_epochs):
        # train
        t0 = time()
        for batch in range(batch_per_epoch):
            optimizer_ortho.zero_grad()
            optimizer_bias.zero_grad()
            x = torch.randn(batch_size, p)
            target = teacher(x)
            pred = student_network(x)
            loss = torch.mean((target - pred) ** 2)
            loss.backward()
            optimizer_ortho.step()
            optimizer_bias.step()
        time_epochs.append(time() - t0)
        # test
        x_test = torch.randn(test_size, p)
        target = teacher(x_test)
        pred = student_network(x_test)
        test_mse = torch.mean((target - pred) ** 2).item()
        test_losses.append(test_mse)
        print(
            "Method %s, time for an epoch : %.1e sec, test MSE: %.2e"
            % (method_name, time_epochs[-1], test_mse)
        )
    plt.semilogy(
        torch.cumsum(torch.tensor(time_epochs), dim=0),
        test_losses,
        label=method_name,
    )
plt.legend()
plt.xlabel("time")
plt.ylabel("test error")
plt.show()
