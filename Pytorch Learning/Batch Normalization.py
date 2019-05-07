# -*- coding:utf-8 -*-
__author__ = 'GIS'
import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 超参数
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = torch.tanh
B_INIT = -0.2

# 训练集
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]  # (-7,10)2000个随机数，np.newaxis 在使用和功能上等价于 None，查看源码发现：newaxis = None，其实就是 None 的一个别名
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise

# 测试集
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)  # 随机数，均值为0，标准差为1，与x相同
test_y = np.square(test_x) - 5 + noise

train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 展示数据
plt.scatter(train_x.numpy(), train_y.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
plt.legend(loc='upper left')
plt.show()

# 搭建神经网络

class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)  # s输入的批量标准化，输入大小为1，调整参数为0.5

        for i in range(N_HIDDEN):
            input_size = 1 if i == 0 else 10   # 输入第一层大小为1 ，后面与隐藏层节点相同为10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)      # 相当于10个 self.fc = fc
            self._set_init(fc)                  # 初始化fc的值
            self.fcs.append(fc)                 # 添加到列表中
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i', bn)
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)         # 每层有10个神经元， 输出变为1个
        self._set_init(self.predict)            # 对输出层，一个设置

    def _set_init(self, layer):                # 初始化函数
        init.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, B_INIT)

    def forward(self, x):
        pre_activation = [x]                    # 还没有经过激活函数时
        if self.do_bn: x = self.bn_input(x)      # 对数据进行标准化处理
        layer_input = [x]                       # 存储每层layer的输入
        for i in range(N_HIDDEN):               # 对每层处理
            x = self.fcs[i](x)
            pre_activation.append(x)            # 处理之前的数据存下来
            if self.do_bn: x = self.bns[i](x)   # 选取一个标准化函数，对x标准化
            x = ACTIVATION(x)                    # 对X进行激活操作
            layer_input.append(x)               # 激活后，x的样子
        out = self.predict(x)                   # 输出处理
        return out, layer_input, pre_activation


nets = [Net(batch_normalization=False), Net(batch_normalization=True)]  # 一个标准化，一个不标准化
print(*nets)

opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

loss_func = torch.nn.MSELoss()


# 可视化
def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5);ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359');ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]: a.set_yticks(());a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct');axs[2, 0].set_ylabel('Act');axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)


if __name__ == '__main__':
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()
    plt.show()

    # 训练过程
    losses = [[], []]

    for epoch in range(EPOCH):
        print('epoch:', epoch)

        layer_inputs, pre_acts = [], []

        for net, l in zip(nets, losses):
            net.eval()
            pred, layer_input, pre_act = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()
        plot_histogram(*layer_inputs, *pre_acts)

        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, opts):   # 对每个网络训练
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()                      # 也学习批量标准化的参数
    plt.ioff()

    # 显示训练损失
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='best')


    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()