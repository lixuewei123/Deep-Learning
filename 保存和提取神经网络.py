# -*- coding:utf-8 -*-
__author__ = 'GIS'

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor),shape = (100,1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor) , shape = (100,1)
x, y = Variable(x), Variable(y)

def save():
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot
    plt.figure(1, figure=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 2 ways to save the net
    torch.save(net1, 'net.pk1')  # save entire network
    torch.save(net1.state_dict(), 'net.params.pk1')  # save only paramaters

def restore_net():
    # restore net1 to net2
    net2 = torch.load('net1')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)