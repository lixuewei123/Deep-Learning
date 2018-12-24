# -*- coding:utf-8 -*-
__author__ = 'GIS'
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#制作伪数据
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)#unsqueeze,将一维的数据变为二维
y = x.pow(2) + 0.2*torch.rand(x.size())#x的2次方，加上一些噪点的影响

x,y = Variable(x),Variable(y)

#打印散点图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


#定义我们的神经网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):#搭建层所需要的一些信息
        super(Net,self).__init__()#继承Net的init模块
        self.myhidden = torch.nn.Linear(n_feature,n_hidden)#定义我们的隐藏层，输入输出
        self.mypredict = torch.nn.Linear(n_hidden,n_output)#定义输出层，输入，输出

    def forward(self, x):#传递的过程，把前面的信息一个个放在forward组合。流程图
        x = F.relu(self.myhidden(x))#activition function
        x = self.mypredict(x)#output 层 调用
        return x

net = Net(1,10,1)
print(net)

# 优化神经网络
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)#随机梯度下降，学习率0.5
loss_func = torch.nn.MSELoss()#损失函数采用均方误差

plt.ion()#实时打印

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()   # 为下次训练清除梯度
    loss.backward()     # 误差反向传播，计算梯度
    optimizer.step()    # 应用梯度

    if t%5 == 0:
        # 可视化学习过程，每5次
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f'% loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()