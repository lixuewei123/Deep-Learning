# -*- coding:utf-8 -*-
__author__ = 'GIS'

import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

N_SAMPLES = 20
N_HIDDEN = 300

# TRAINING DATA
x = torch.linspace(-1, 1, N_SAMPLES)  # 20个-1到1之间的随机数，一维向量
x = torch.unsqueeze(x, 1)  # 参数1 ，将一个横向量变成了纵向量
# print(x)
# print(torch.numel(x))
# print(x.size())

y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))  # 从标准差为1，均值为0，正态分布中抽取随机数
# print(y)
# print(y.size())
# print(torch.zeros(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# 展示数据
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 搭建神经网络
# 我们在这里搭建两个神经网络, 一个没有 dropout, 一个有 dropout.
# 没有dropout
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),  # 输入为20*1的向量
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),  # 300 * 300
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # 忽略0.5的神经元
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

print('overfitting:', net_overfitting)
print('drop:', net_dropped)

# 训练的时候, 这两个神经网络分开训练. 训练的环境都一样.
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()


plt.ion()  # something about plotting
for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)

    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        net_overfitting.eval()  # 在预测时，需要把dropout取消掉，.eval(）进入预测模式
        net_dropped.eval()

        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)
        # change back to train mode
        net_overfitting.train()
        net_dropped.train()

plt.ioff()
plt.show()