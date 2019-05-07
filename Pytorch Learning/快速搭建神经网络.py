# -*- coding:utf-8 -*-
__author__ = 'GIS'

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 制造伪数据
n_data = torch.ones(100, 2)  # 100*2
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)  # 创建一个全为1的张量
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承Net的init模块
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, tensor_x):
        tensor_x = F.relu(self.hidden(tensor_x))
        tensor_x = self.output(tensor_x)
        return tensor_x


net1 = Net(2, 10, 2)
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print("net1 is:", net1)
print("net2 is:", net2)
