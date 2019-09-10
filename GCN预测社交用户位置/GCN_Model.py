# -*- coding:utf-8 -*-
__author__ = 'GIS'

import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from GCN_Layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, batch_normalization=False, highway=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.do_bn = batch_normalization        # 是否层标准化
        self.bn_input_gc1 = nn.BatchNorm1d(num_features=nhid, momentum=0.1)
        self.bn_input_gc2 = nn.BatchNorm1d(num_features=nhid, momentum=0.1)
        self.highway = highway

    def forward(self, x, adj):
        x = self.gc1(x, adj)                                    # gcn卷积
        if self.do_bn: x = self.bn_input_gc1(x)                 # BN
        x = F.relu(x)                                           # relu激活
        # if self.highway:
        #     x = highway_dense(x)                                # highway_dense层
        x = F.dropout(x, self.dropout, training=self.training)  # dropout
        x = self.gc2(x, adj)
        if self.do_bn: x = self.bn_input_gc2(x)                 # 第二层gcn的BN
        return F.log_softmax(x, dim=1)


class Highway_gate_gcn(Module):
    def __init__(self, nfeat, nhid, output, dropout, batch_normalization=False, highway=True):
        super(Highway_gate_gcn, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.do_bn = batch_normalization        # 是否层标准化
        self.bn_input_gc = nn.BatchNorm1d(num_features=nhid, momentum=0.1)
        self.full_connect = nn.Linear(in_features=nhid, out_features=nhid)
        self.highway = highway
        self.gc3 = GraphConvolution(nhid, output)

    def forward(self, x, adj):
        x = self.gc1(x, adj)     # GCN卷积  9475,100
        if self.do_bn: x = self.bn_input_gc(x)  # BN = False
        x = F.tanh(x)               # 非线性激活
        x = F.dropout(x, self.dropout, training=self.training)  # dropout
        h1 = x                      # incoming     9475,100
        x = self.gc2(x, adj)        # 第二层GCN   9475,100
        if self.do_bn: x = self.bn_input_gc(x)  # BN = False
        x = F.sigmoid(x)            # sigmoid激活     9475,100
        h2 = x                      # l_h           9475,100
        h3 = F.sigmoid(self.full_connect(h1))    # l_t  9475,100
        output = h3 * h2 + (1.0 - h3) * h1
        output = F.log_softmax(self.gc3(output, adj))
        return output







def accuracy(output, labels):
    # output 2708*7 按照行选取最大值2708个 ,提取2708个最大值的位置（0-7）
    preds = output.max(1)[1].type_as(labels)  # 2708维，表示每个节点的分类位置
    correct = preds.eq(labels).double()     # preds 2708维，labels 2708维  对比pred-labels对应位置的值对不对
    correct = correct.sum()                 # 预测正确的数量
    return correct / len(labels)            # 返回预测正确率


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)