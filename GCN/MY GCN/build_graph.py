# -*- coding:utf-8 -*-
__author__ = 'GIS'

import networkx as nx
import dgl
import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
import torch


def read_nodes():
    with open("E:/deep_network/deal_data/regulation data/nodes10w.txt", 'r') as f:
        nodes_list1 = f.readlines()
    nodes_list2 = []
    for node in nodes_list1:
        node = re.sub('\n', '', node)
        nodes_list2.append(node)
    return nodes_list2


def read_edges():
    with open("E:/deep_network/deal_data/regulation data/edges36w .txt", 'r') as f:
        edges_list1 = f.readlines()

    i = 0
    edges = []
    for ed in edges_list1:
        i += 1
        # print(i)
        ed = re.sub("[\(\)\'\ ]", '', ed).strip()
        ed = ed.split(',')
        edges.append(ed)

    edges_df = pd.DataFrame(edges)
    return edges_df


def edges_dataframe(nodes, edges_df):
    node_map = {j: i for i, j in enumerate(nodes)}
    e1 = edges_df[0]
    e2 = edges_df[1]
    e11 = list(map(node_map.get, e1))
    e22 = list(map(node_map.get, e2))
    edges_df[2] = e11
    edges_df[3] = e22
    return edges_df


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
        G.add_edges(edges_weight['s'][i], edges_weight['e'][i], {'w': edges_weight['count'][i]})

    G.add_edges(e5, {'w': np.array(list(edges_weight['count']))})
    print('节点数', len(G.nodes), '边数', len(G.edges))


def build_graph_adj(edges_df, nodes):
    #==================================================================
    #                       添加权重
    #==================================================================
    e1 = edges_df[2]    # 起始边序号
    e2 = edges_df[3]    # 结束边序号
    # 计算权重
    # e3 = []
    # for i, j in zip(e1, e2):
    #     i = int(i)
    #     j = int(j)
    #     if i > j:
    #         e3.append((i, j))
    #     else:
    #         e3.append((j, i))
    e3 = []
    for i, j in zip(e1, e2):
        e3.append((i, j))

    edges_df[5] = e3    # 元组表示一条边
    edges_weight = pd.DataFrame(edges_df[5].value_counts())
    edges_weight[1] = edges_weight.index
    edges_weight[1][0][0]
    edges_weight.columns = ['count', 'edge']
    print('权重边', edges_weight.shape)
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
    edges_df = edges_weight
    weight = np.array(edges_df['count'])    # 权重
    edges = np.array(edges_df[['s', 'e']])  # 边
    adj = sp.coo_matrix((np.array(edges_df['count']), (edges[:, 0], edges[:, 1])),
                        shape=(len(nodes), len(nodes)),
                        dtype=np.float32)
    #========================================================================

    #----------------没有添加权重---------------------------------------------
    # edges = np.array(edges_df[[2, 3]])
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(len(nodes), len(nodes)),
    #                     dtype=np.float32)
    #-------------------------------------------------------------------------
    # degrees = []
    # for i in range(len(nodes)):
    #     print('计算度矩阵', i)
    #     start = edges_weight[edges_weight['s'] == i]
    #     end = edges_weight[edges_weight['e'] == i]
    #     start = start.shape[0]
    #     end = end.shape[0]
    #     degree = start + end
    #     degrees.append(degree)
    # degrees = np.array(degrees)*-1/2
    # np.save('E:/deep_network/deal_data/regulation data/degrees.npy', degrees)
    # degrees = np.load('E:/deep_network/deal_data/regulation data/degrees.npy')
    # D = sp.coo_matrix((np.array(degrees), (range(len(nodes)), range(len(nodes)))),
    #                     shape=(len(nodes), len(nodes)),
    #                     dtype=np.float32)
    # adj = D*adj*D
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print(adj.shape)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # tocoo([copy])：返回稀疏矩阵的coo_matrix形式
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 堆叠数组,coo_matrix.row数据的行号，coo_matrix.col数据的列号
    values = torch.from_numpy(sparse_mx.data)  # coo矩阵转tensor
    shape = torch.Size(sparse_mx.shape)  # 系数矩阵的形状
    return torch.sparse.FloatTensor(indices, values, shape)  # 返回tensor的稀疏矩阵


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def dot(x, y, sparse=False):  # 定义矩阵的乘法，如果是稀疏的或者不是
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv).tocoo()
    return mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = rowsum.astype(np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def get_labels(nodes):
    df = pd.read_csv("E:/deep_network/deal_data/regulation data/LXW/users40w.csv", sep=',', header=None,
                     encoding='utf=8')
    df.columns = ['ID', 'name', 'type', 'url', 'guanzhu', 'fensi', 'weibo', 'xingbie',
                  'address', 'jianjie', 'biaoqian', 'jiaoyu']
    s = list(df['address'])
    s1 = []
    for ele in s:
        ele = ele[0:2]
        s1.append(ele)
    df['address2'] = s1
    df = df[df['name'].isin(nodes)]
    df = df[['name', 'address2']]
    print('标签属性数目', df.shape)
    # 添加标签
    labels = []
    for i, node in enumerate(nodes):
        print('第{}个标签'.format(i))
        s = df[df['name'] == node]
        s = list(s['address2'])[0]
        labels.append(s)
    return labels


def other_meta_feature():
    df = pd.read_csv("E:/deep_network/deal_data/regulation data/LXW/users40w.csv", sep=',', header=None,
                     encoding='utf=8')
    df.columns = ['ID', 'name', 'type', 'url', 'guanzhu', 'fensi', 'weibo', 'xingbie', 'address', 'jianjie', 'biaoqian',
                  'jiaoyu']
    df = df[df['type'] != '未找到']
    df = df[df['type'] == '普通用户']
    s = list(df['address'])
    s1 = []
    for ele in s:
        ele = ele[0:2]
        s1.append(ele)
    df['address2'] = s1
    df = df[df['address2'] != '其他']  # 删除没有位置的
    df = df[df['address2'] != '海外']  # 删除海外的
    df.reset_index(drop=True)
    guanzhu = []
    fensi = []
    weibo =[]
    xingbie = []
    for i, node in enumerate(nodes):
        print('第 {} 个标签'.format(i))
        s = df[df['name'] == node]
        s0 = list(s['fensi'])[0]
        guanzhu.append(s0)
        s1 = list(s['weibo'])[0]
        weibo.append(s1)
        s2 = list(s['guanzhu'])[0]
        guanzhu.append(s2)
        s3 = list(s['xingbie'])[0]
        xingbie.append(s3)
        return guanzhu, fensi, weibo, xingbie


def mainly():
    nodes = read_nodes()
    edges_df = read_edges()
    edges_df = edges_dataframe(nodes, edges_df)
    # edges_df.to_csv("E:/deep_network/deal_data/regulation data/edges_df.csv", header=None,
    #                 index=False, encoding='utf-8', sep=',')
    adj = build_graph_adj(edges_df, nodes)
    adj = normalize(adj)  # 邻接矩阵--->归一化拉普拉斯矩阵
    # adj = normalize(adj + sp.eye(adj.shape[0]))  # adj + 单位矩阵I 再标准化

    # 使用位置特征
    # features = get_labels(nodes)
    # features = encode_onehot(features)
    # labels = np.where(features)[1]  # 运行产生标签
    features = np.load('E:/deep_network/deal_data/regulation data/features.npy')
    features = preprocess_features(features)    # 特征矩阵行标准化
    labels = np.load('E:/deep_network/deal_data/regulation data/label.npy')  # 直接读取

    # 消除部分节点的位置属性测试
    # features = features[0:80000]
    # test = np.zeros((11669, 34))  # 最后1w多数据属于缺失用户，无位置特征，只有邻居特征
    # features = np.concatenate((features, test), axis=0)
    print(features.shape)

    # 使用元数据特征
    # guanzhu = np.load('E:/deep_network/deal_data/regulation data/guanzhu.npy')
    # fensi = np.load('E:/deep_network/deal_data/regulation data/fensi.npy')
    # weibo = np.load('E:/deep_network/deal_data/regulation data/weibo.npy')
    # xingbie = np.load('E:/deep_network/deal_data/regulation data/xingbie.npy')
    # print(guanzhu.shape)
    # guanzhu = guanzhu.reshape(guanzhu.shape[0], 1)
    # fensi = fensi.reshape(fensi.shape[0], 1)
    # weibo = weibo.reshape(weibo.shape[0], 1)
    # features = np.concatenate((features, xingbie), axis=1)
    # features = xingbie
    # features = np.zeros([91669, 34])
    # features = np.concatenate((features, guanzhu, fensi, weibo), axis=1)
    print('特征矩阵', features.shape)
    return adj, features, labels





# if __name__ == '__main__':
#     main()