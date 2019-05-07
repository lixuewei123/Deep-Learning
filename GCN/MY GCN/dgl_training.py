# -*- coding:utf-8 -*-
__author__ = 'GIS'

import networkx as nx
import dgl
import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
import torch
import torch
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx
import sys
sys.path.append("E:/deep_network/deal_data/regulation code")
from build_graph import read_nodes
from build_graph import read_edges
from build_graph import edges_dataframe
from build_graph import mainly


def data_ready():
    nodes = read_nodes()
    edges_df = read_edges()
    edges_df = edges_dataframe(nodes, edges_df)
    print('nodes', len(nodes))
    print('edges', edges_df.shape)
    return nodes, edges_df


def build_graph(edges_df, nodes):
    G = dgl.DGLGraph()
    G.add_nodes(len(nodes))
    print('节点数', len(G.nodes), '边数', len(G.edges))

    e111 = list(edges_df[2])
    e222 = list(edges_df[3])
    e333 = []
    for i, j in zip(e111, e222):
        e333.append((i, j))
    edges_df[4] = e333
    edges_weight = pd.DataFrame(edges_df[4].value_counts())
    edges_weight[2] = edges_weight.index
    edges_weight.columns = ['count', 'edge']

    e5 = []
    for ele in list(edges_weight['edge']):
        e5.append(list(ele))

    e6 = []
    e7 = []
    for ele in e5:
        e6.append(ele[0])
        e7.append(ele[1])
    edges_weight['s'] = e6
    edges_weight['e'] = e7

    for i in range(len(e5)):
        G.add_edge(edges_weight['s'][i], edges_weight['e'][i])

    print('节点数', len(G.nodes), '边数', len(G.edges))
    return G


def gcn_message(edges):
    return {'msg': edges.src['fea']}  # 156*34


def gcn_reduce(nodes):
    return {'fea': torch.sum(nodes.mailbox['msg'], dim=1)}  # 更新节点的h属性为


msg_func = fn.copy_src(src='h', out='m')
reduce_func = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):  # github
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg_func, reduce_func)
        g.apply_nodes(func=self.apply)
        return g.ndata.pop('h')


class GCNLayer_1(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):  # 输入是图g, 和inputs即特征矩阵
        g.ndata['fea'] = inputs   # 为图g添加特征h
        g.send(g.edges(), gcn_message)   # 给定边发送消息，发送信息， 发送消息的函数是
        g.recv(g.nodes(), gcn_reduce)  # nodes接受邻居消息，聚合信息
        fea = g.ndata.pop('fea')  # h作为节点特征，通过节点间的发送信息，聚合信息，发生了改变
        # print(fea.shape)
        return self.linear(fea)


# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)  # 第一层的in_feats就是输入的特征矩阵的维度，hidden_size是变换后的维度
        self.gcn2 = GCNLayer(hidden_size, num_classes)  # num_calss是类别数量

    def forward(self, g, inputs):  # 输入是图g， 和node特征矩阵
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


def accuracy(output, labels):
    # output 10886*2 按照行选取最大值2708个 ,提取2708个最大值的位置（0-7）
    preds = output.max(1)[1].type_as(labels)  # 2708维，表示每个节点的分类位置
    correct = preds.eq(labels).double()     # preds 2708维，labels 2708维  对比pred-labels对应位置的值对不对
    correct = correct.sum()                 # 预测正确的数量
    return correct / len(labels)            # 返回预测正确率


def main():
    nodes, edge_df = data_ready()
    G = build_graph(edge_df, nodes)
    adj, features, labels = mainly()
    inputs = features
    net = GCN(34, 68, 34)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)



    # training
    inputs = torch.FloatTensor(np.array(inputs))
    labels = torch.LongTensor(labels)
    # net.cuda()
    # inputs.cuda()
    # labels.cuda()
    # G.cuda()

    # 训练优化
    train = torch.tensor(range(0, 60000))
    val = torch.tensor(range(60000, 70000))
    test = torch.tensor(range(70000, 90000))

    for epoch in range(1000):
        output = net(G, inputs)
        logp = F.log_softmax(output, 1)
        # print('softmax激活后：', logp.shape, 'labels:', labels.shape)

        # loss = F.nll_loss(logp, labels)
        # 半监督，删除部分节点
        loss = F.nll_loss(logp[train], labels[train])

        acc = accuracy(output[train], labels[train])
        acc_val = accuracy(output[val], labels[val])
        acc_test = accuracy(output[test], labels[test])  # 测试节点


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'val: {:.4f}'.format(acc_val.item()),
              'test: {:.4f}'.format(acc_test.item()))


if __name__ == '__main__':
    main()