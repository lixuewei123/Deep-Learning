# -*- coding:utf-8 -*-
__author__ = 'GIS'

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#制作伪数据
n_data = torch.ones(100, 2)  # 100*2
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)  # 创建一个全为1的张量
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# 打印散点图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# 定义我们的神经网络


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 搭建层所需要的一些信息
        super(Net, self).__init__()  # 继承Net的init模块
        self.myhidden = torch.nn.Linear(n_feature, n_hidden)  # 定义我们的隐藏层，输入输出
        self.mypredict = torch.nn.Linear(n_hidden, n_output)  # 定义输出层，输入，输出

    def forward(self, x):  # 传递的过程，把前面的信息一个个放在forward组合。流程图
        x = F.relu(self.myhidden(x))  # activition function
        x = self.mypredict(x)  # output 层 调用
        return x


net = Net(2,10,2)
print(net)

# 优化神经网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 随机梯度下降，学习率0.5
loss_func = torch.nn.CrossEntropyLoss()  # 在分类的时候使用这个损失函数，计算softmax,即概率

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()