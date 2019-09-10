# -*- coding:utf-8 -*-
__author__ = 'GIS'

import matplotlib
matplotlib.use('Agg')
import os
import torch.optim as optim
import sys
import torch
import argparse
import pickle
import pdb
import copy
import logging
import time
import json
import numpy as np
from haversine import haversine
import gzip
import codecs
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import json
import re
import networkx as nx
import scipy as sp
import random
import argparse
import sys
from collections import Counter
from GCN_Model import GCN
from GCN_Model import Highway_gate_gcn
from Graph_Class import Graph_class
import torch
import pdb
import pickle
import hickle

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG,
                    filename="E:\\deep_network\\GCN Highway My Data\\GCN_train.log",
                    filemode='w+')
logging.info("开始运行。。。")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


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


def dump_obj(obj, filename, protocol=-1, serializer=pickle):  # 序列化对象，将对象obj保存到文件file中去。
    if serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')  # 写入filename
    else:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=pickle):  # 反序列化对象，将文件中的数据解析为一个python对象。
    if serializer == hickle:
        obj = serializer.load(filename)
    else:
        with gzip.open(filename, 'rb') as fin:
            obj = serializer.load(fin)
    return obj


class training():
    def __init__(self, home_data, model_args = None, encoding='utf-8', celebrity_threshold=15, mindf=10,
                 dtype='float32', one_hot_label=False, model=None, dropout=0.0, EPOCH=50, guli=False):
        self.home_data = home_data
        self.model_args = model_args
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.mindf = mindf
        self.dtype = dtype
        self.one_hot_label = one_hot_label
        self.model = model
        self.dropout = dropout
        self.EPOCH = EPOCH
        self.guli = guli


    def process_data(self):
        model_args = self.model_args
        home_data = self.home_data
        encoding = self.encoding
        celebrity_threshold = self.celebrity_threshold
        mindf = self.mindf
        dtype = self.dtype
        one_hot_label = self.one_hot_label
        logging.info('loading data from dumped file...')
        vocab_file = os.path.join(home_data, 'vocab.pkl')
        dump_file = os.path.join(home_data, 'dump.pkl')
        if os.path.exists(dump_file) and not model_args.builddata:
            print("process_data使用预加载的数据")
            logging.info('loading data from dumped file...')
            data = load_obj(dump_file)  # 调用载入函数,载入的data就包含了13个元素data:拉普拉斯矩阵A（9475,9475,211451），训练集（5685,9467,647167）和训练集标签(5685)、验证集(1895,9467,214920)和验证集标签(1895)，测试集(1895,9467,215916)和测试集标签(1895)，训练集用户(5685个用户名list)、验证集用户、测试集用户、lat中位数32个、lng中位数32个、所有用户9475的坐标
            logging.info('loading data finished!')
            self.data = data
            return data  # 到这一步函数结束，返回了处理好的data

        data_loader = Graph_class(home_data=home_data, input_df=input_df, colname=colname,
                                  encoding=encoding, celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label,
                                  mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b', docsize=5000)
        data_loader.load_data()
        print("载入数据成功，开始建模图")
        data_loader.get_graph()
        if self.guli:
            data_loader.delete_guli_nodes()
        nx.write_adjlist(data_loader.graph, home_data + 'graph.dat')
        print("图建模成功，#nodes: %d, #edges: %d" % (nx.number_of_nodes(data_loader.graph), nx.number_of_edges(data_loader.graph)))
        data_loader.tfidf()
        print("tfidf向量转换成功", data_loader.X.shape)
        # data_loader.doc2vec()
        # print("doc2vec向量转换成功", data_loader.d2v.shape)
        data_loader.assignlabel()

        vocab = data_loader.vectorizer.vocabulary_  # 语料库词典
        logging.info('saving vocab in {}'.format(vocab_file))
        dump_obj(vocab, vocab_file)  # 载入语料库数据

        users = data_loader.users
        logging.info('creating adjacency matrix...')
        adj = nx.adjacency_matrix(data_loader.graph, nodelist=range(len(users)), weight='w')  # 获取邻接矩阵
        adj.setdiag(0)              # 对角线元素设置为0
        selfloop_value = 1
        adj.setdiag(selfloop_value)  # 对角线元素设置为1
        logging.info('creating adjacency matrix shape %s' % str(adj.shape))
        n, m = adj.shape  # adj的维度
        diags = adj.sum(axis=1).flatten()  # 返回每列的和，向量
        with sp.errstate(divide='ignore'):
            diags_sqrt = 1.0 / sp.sqrt(diags)
        diags_sqrt[sp.isinf(diags_sqrt)] = 0
        D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
        A = D_pow_neghalf * adj * D_pow_neghalf
        A = A.astype(dtype)  # 得到拉普拉斯矩阵A
        logging.info('adjacency matrix created. shape: %s' % str(A.shape))
        X = data_loader.X
        # X = data_loader.d2v
        Y = data_loader.Y

        data = (A, X, Y)
        self.data = data
        if not model_args.builddata:
            logging.info('dumping data in {} ...'.format(str(dump_file)))
            dump_obj(data, dump_file)
            logging.info('data dump finished!')
        return data


    def main(self):
        model = self.model
        logging.info("采用 %s 模型计算" % str(model))
        print("模型采用 ：%s" % model)
        batch_size = 500  # batch_size=500
        hidden_size = [100]  # 隐层节点数100
        dropout = self.dropout  # dropout=0
        regul = 1e-6  # 正则化系数0.000001
        dtype = 'float32'
        dtypeint = 'int32'
        lr = 0.01
        check_percentiles = False
        A, X, Y = self.data
        logging.info("载入数据：拉普拉斯矩阵A，tfidf特征X，类标签Y %s"% str(str(A.shape) + str(X.shape) + str(Y.shape)))

        input_size = X.shape[1]  # 输入大小就是X的列数，也是特征数
        output_size = np.max(Y) + 1  # 输出规模是Y的最大值+1
        rows = X.shape[0]
        idx_train = range(int(rows*0.6))
        idx_val = range(int(rows*0.6), int(rows*0.8))
        idx_test = range(int(rows*0.8), rows)


        # data
        X = sparse_mx_to_torch_sparse_tensor(X)
        # X = torch.Tensor(X)
        A = sparse_mx_to_torch_sparse_tensor(A)
        Y = torch.LongTensor(Y)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # GCN model
        if model == 'GCN':
            print("采用GCN模型计算")
            logging.info("model参数设置： nfeat %d   nhid %d   nclass %d  dropout %f" % (input_size, 500, output_size, dropout))
            clf = GCN(nfeat=input_size, nhid=800, nclass=output_size, dropout=dropout, batch_normalization=False,
                      highway=False)
        else:
            print("采用Highway_gate_gcn模型计算")
            logging.info("model参数设置： nfeat %d   nhid %d   nclass %d  dropout %f" % (input_size, input_size, output_size, dropout))
            clf = Highway_gate_gcn(nfeat=input_size, nhid=hidden_size[0], output=output_size, dropout=dropout,
                                   batch_normalization=False, highway=True)
        optimizer = optim.Adam(clf.parameters(),  # Adam优化
                               lr=lr, weight_decay=regul)

        clf.cuda()
        X = X.cuda()
        A = A.cuda()
        Y = Y.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        print(X.shape, A.shape, Y.shape)

        EPOCH = self.EPOCH
        logging.info("迭代次数为 %d" % EPOCH)
        for epoch in range(EPOCH):
            t = time.time()
            clf.train()
            optimizer.zero_grad()
            output = clf(X, A)
            loss_train = F.nll_loss(output[idx_train], Y[idx_train])
            acc_train = accuracy(output[idx_train], Y[idx_train])
            loss_train.backward()
            optimizer.step()

            loss_val = F.nll_loss(output[idx_val], Y[idx_val])  # 验证集的损失
            acc_val = accuracy(output[idx_val], Y[idx_val])  # 验证集的准确度
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        clf.eval()  # 腾出内存
        output = clf(X, A)
        loss_test = F.nll_loss(output[idx_test], Y[idx_test])
        acc_test = accuracy(output[idx_test], Y[idx_test])
        logging.info("训练集、验证集、测试集划分： 0-38224-50965-63706")
        logging.info("Test set results: loss= {: %.4f} accuracy= {: %.4f}" % (loss_test.item(), acc_test.item()) )
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', metavar='str', help='dataset for dialectology', type=str, default='na')
    parser.add_argument('-bucket', '--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-batch', '--batch', metavar='int', help='SGD batch size', type=int, default=500)
    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument('-mindf', '--mindf', metavar='int', help='minimum document frequency in BoW', type=int,
                        default=10)
    parser.add_argument('-d', '--dir', metavar='str', help='home directory', type=str,
                        default='E:/deep_network/文档/模型框架/GCN Highway Gate General Framework/data')
    parser.add_argument('-enc', '--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str,
                        default='utf-8')
    parser.add_argument('-reg', '--regularization', metavar='float', help='regularization coefficient)', type=float,
                        default=1e-6)
    parser.add_argument('-cel', '--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument('-conv', '--convolution', action='store_true', help='if true do convolution')
    parser.add_argument('-tune', '--tune', action='store_true', help='if true tune the hyper-parameters')
    parser.add_argument('-tf', '--tensorflow', action='store_true', help='if exists run with tensorflow')
    parser.add_argument('-batchnorm', action='store_true', help='if exists do batch normalization')
    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)
    parser.add_argument('-percent', action='store_true', help='if exists loop over different train/dev proportions')
    parser.add_argument('-vis', metavar='str', help='visualise representations', type=str, default=None)
    parser.add_argument('-builddata', action='store_true',
                        help='if exists do not reload dumped data, build it from scratch')
    parser.add_argument('-lp', action='store_true', help='if exists use label information')
    parser.add_argument('-notxt', action='store_false', help='if exists do not use text information')
    parser.add_argument('-maxdown', help='max iter for early stopping', type=int, default=10)
    parser.add_argument('-silent', action='store_true', help='if exists be silent during training')
    parser.add_argument('-highway', action='store_true', help='if exists use highway connections else do not')
    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-save', action='store_true', help='if exists save the model after training')
    parser.add_argument('-load', action='store_true', help='if exists load pretrained model from file')
    parser.add_argument('-feature_report', action='store_true',
                        help='if exists report the important features of each location')
    parser.add_argument('-lblfraction', nargs='+', type=float,
                        help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"
    input_df = "16w_user_340wweibo_merge_province_label164016_delaite.csv"
    colname = ['content', 'label', '用户ID', '用户名', 'aite', 'content_all']
    input_df = "837908uid_province_content_aite.csv"
    colname = ['aite', 'content', 'label', '用户ID']

    args = parse_args(sys.argv[1:])
    model_args = args

    train = training(home_data=home_data, model_args=model_args, dropout=0.5, EPOCH=12,
                     celebrity_threshold=8, mindf=8, model='GCN', guli=True)
    data = train.process_data()
    # pdb.set_trace()
    train.main()