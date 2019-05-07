# -*- coding:utf-8 -*-
__author__ = 'GIS'
import torch
import numpy as np
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)
print(tensor)
print(variable)
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)
#variable 的反向传递
v_out.backward()#对v_out 反向传播，variable的值也会受影响，因为他们是一套体系中的
print("反向传播梯度")
print(variable.grad)#variable的梯度
#variable的结果怎么来的呢？
# v_out = 1/4*sum(variable*variable)
# d(v_out)/d(variable) = 1/4*2*variable = variable/2      2是平方      d()梯度

#看一下variable有哪些属性
print("variable的各种属性")
print(variable)
print(variable.data)#tensor格式
print(variable.data.numpy())#numpy格式