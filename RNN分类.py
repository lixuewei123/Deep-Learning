# -*- coding:utf-8 -*-
__author__ = 'GIS'

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

# Hyper parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STET = 28  # 以图片为例，28行表示时间
INPUT_SIZE = 28  # 28列代表输入
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(root='./mnist', transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]  # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,  # 1 个 cell
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):   # x shape (batch, time_step, input_size)
            r_out, (h_n, h_c) = self.rnn(x, None)

            out = self.out(r_out[:, -1, :])  # 最后一个时刻的output
            return out




rnn2 = RNN()
print(rnn2)

optimizer = torch.optim.Adam(rnn2.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        # b_x = Variable(b_x)
        # b_y = Variable(b_y)
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        out_put = rnn2(Variable(b_x))
        out_put = rnn2(b_x)             # rnn output
        loss = loss_func(out_put, b_y)  # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_out = rnn2(test_x)
            pred_y = torch.max(test_out, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum())/float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')