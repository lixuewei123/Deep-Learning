# -*- coding:utf-8 -*-
__author__ = 'GIS'

import sys
sys.path.append("E:\\lxw_data\\GCN")

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from build_graph import mainly
from build_graph import sparse_mx_to_torch_sparse_tensor
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # 1/sqrt(weight列数)
        self.weight.data.uniform_(-stdv, stdv)  # 生成均匀分布（-0.5,0.5）按照weight的shape
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # 生成均匀分布（-0.5,0.5）bias的形状

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def accuracy(output, labels):
    # output 10886*2 按照行选取最大值2708个 ,提取2708个最大值的位置（0-7）
    preds = output.max(1)[1].type_as(labels)  # 2708维，表示每个节点的分类位置
    correct = preds.eq(labels).double()     # preds 2708维，labels 2708维  对比pred-labels对应位置的值对不对
    correct = correct.sum()                 # 预测正确的数量
    return correct / len(labels)            # 返回预测正确率


adj, features, labels = mainly()


# train
adj = sparse_mx_to_torch_sparse_tensor(adj)
features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(labels)

idx_train = range(0, 36369)
idx_val = range(20000, 25000)
idx_test = range(25000, 31000)
idx_train = torch.LongTensor(idx_train)  # 都换成long格式
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = GCN(nfeat=34,   # 输入特征2个特征
            nhid=68,   # 隐藏层，默认16个
            nclass=34,  # 7类
            dropout=0.5)   # dropout训练，默认0.5
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

# 训练500次
for i in range(200):
    for epoch in range(500):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # 输出10886个节点的特征预测
        output = model(features, adj)  # 输入 特征矩阵10886*2  拉普拉斯矩阵10886*10886  输出10886*n
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 选140个输出样本和标签计算损失
        acc_train = accuracy(output[idx_train], labels[idx_train])  # 返回140个中预测的准确率

        loss_train.backward()   # 误差返向传播
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 验证集的损失
        acc_val = accuracy(output[idx_val], labels[idx_val])  # 验证集的准确度

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])  # 测试集的损失
        acc_test = accuracy(output[idx_test], labels[idx_test])  # 测试集的准确度

        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()),
        #       'loss_test: {:.4f}'.format(loss_test.item()),
        #       'acc_test: {:.4f}'.format(acc_test.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

    print(i)
    yuce = output.max(1)[1]
    yuce = np.array(yuce.cpu())
    np.save('E:\\lxw_data\\data\\regulation data\\test\\yuce--'+str(i)+'.npy', yuce)
    accc = output.max(1)[1].type_as(labels).eq(labels)
    accc = np.array(accc.cpu())
    np.save('E:\\lxw_data\\data\\regulation data\\test\\acc--'+str(i)+'.npy', accc)