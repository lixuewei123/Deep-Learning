# -*- coding:utf-8 -*-
__author__ = 'GIS'
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Hyper parameters
EPOCH = 1  # 训练时间比较长所以就1次
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 将原始的数据转成tensor  （0,1）  <----  (0,255)
                                                  # torchvision.transforms.ToTensor将一个图片转成张量的形式
    download=DOWNLOAD_MNIST,
)

# plot
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# print(type(train_data))
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.  #  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]  # 前2000个


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 卷积层
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,  # 图片有多少层，灰度图片1层
                out_channels=16,  # 有多少个输出高度，filter的个数,16个卷积核提取16个特征
                kernel_size=5,  # filter长宽都是5个像素点
                stride=1,  # 步长
                padding=2,  # 填充0，保持卷积图像大小不变。 if stride=1,padding = (kernel_size-1)/2 = (5-1)/2
            ),  # (16, 28, 28)
            nn.ReLU(),  # (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # 长和宽为2的池化 (16, 14, 14)
        )
        self.conv2 = nn.Sequential(      # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32, 14, 14)
            nn.ReLU(),                   # (32, 14, 14)
            nn.MaxPool2d(2)              # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)    # 10类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7)--->(batch,32 * 7 * 7)展平
        output = self.out(x)       # (batch,32 * 7 * 7)
        return output   # 输出类别


cnn = CNN()
# print(cnn)  # 打印网络结构

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # 使用label数据

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # print(step)
        # print(type(x))
        # print(x)
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            print('pred_y:', type(pred_y))
            print('test_y:', type(test_y))
            print('test_y.size=', test_y.size(0))
            accuracy = sum(pred_y.data.numpy() == test_y.data.numpy()) / test_y.size(0)
            print(accuracy)
            print('EPoch:', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.4f' % accuracy)


# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')