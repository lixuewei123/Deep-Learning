# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import numpy as np
import networkx as nx
import re
import os
import sys
import pickle
import collections
import logging
import pdb
import gensim
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
import random


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO,
                    filename="E:/deep_network/GCN Highway My Data/Graph_class.log",
                    filemode='w+')

def efficient_collaboration_weighted_projected_graph2(B, nodes):  # 输入图g， 0:60000
    nodes = set(nodes)          # nodes去重复，0，1,2,3...60000
    G = nx.Graph()
    G.add_nodes_from(nodes)		# 添加60000多个节点ID 节点为0-60000
    all_nodes = set(B.nodes())  # B图中所有的节点，60000多个用户 + @用户的非名人节点 0-6w+
    i = 0
    tenpercent = len(all_nodes) / 10  # 每10次，日志记录一次
    for m in all_nodes:			# g图中所有的节点，包括60000多个用户和@用户
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  # 记录到第i个节点了

        nbrs = B[m]  # 可以看在g图中，m节点所有的邻居
        target_nbrs = [t for t in nbrs if t in nodes]  # 如果m的邻居都在nodes（60000用户）中，就添加到列表target_nbrs
        if m in nodes:  			# 如果m节点也在nodes中
            for n in target_nbrs:  	# m的在nodes中的邻居id， 也就是在nodes中的节点以及邻居，复制他们之间的边
                if m < n:			# 如果m<n， 建立边，没有这个判定条件也无所谓，减少了一半的运算量。
                    if not G.has_edge(m, n):	# 如果m和n之间没有边
                        G.add_edge(m, n)		# 建立无向无权边
		# m的所有在nodes中的邻居之间，也建立边。
        for n1 in target_nbrs:				# 在nodes中，m的邻居用户
            for n2 in target_nbrs:			# 在nodes中，m的其他邻居用户
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G


def labelizeContent(content, label_type):  # 这是常用的加label函数了，你在网上搜也是差不多这样的。
    labelized = []
    for i, v in enumerate(content):
        label = '%s_%s' % (label_type, i)
        labelized.append(gensim.models.doc2vec.TaggedDocument(v, [label])) #TaggedDocument 与 LabeledSentence是一样效果的，后者是基于前者的。
    return labelized


def get_d2v(model_dm, nodes):
    shape = model_dm.docvecs['train_' + '1'].shape[0]
    features_d2v = model_dm.docvecs['train_' + str(0)].reshape(1, shape)
    for i in range(len(nodes)):
        print(i)
        t = model_dm.docvecs['train_' + str(i)].reshape(1, shape)
        features_d2v = np.concatenate((features_d2v, t), axis=0)
    features_d2v = np.delete(features_d2v, 0, axis=0)
    return features_d2v


def getVecs(model, corpus):
    # vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    # return np.concatenate(vecs)
    vecs = []
    for text in corpus:
        tmp = [model[w] for w in text.words]
        tmp = np.array(tmp)
        vecs.append(tmp.sum(axis=0))
    return np.array(vecs)


class Graph_class():
    def __init__(self, home_data, input_df, colname, celebrity_threshold, tokenizer=None,
                 token_pattern=r'(?u)(?<![#@])\b\w\w+\b', idf=True, norm='l2',
                 btf=True, subtf=False, stops=None, vocab=None, encoding='utf-8',
                 one_hot_labels=False, mindf=10, maxdf=0.2, docsize=5000):
        self.home_data = home_data
        self.input_df = input_df
        self.colname = colname
        self.celebrity_threshold = celebrity_threshold
        self.tokenizer = tokenizer
        self.token_pattern = token_pattern
        self.idf = idf
        self.norm = norm
        self.btf = btf
        self.subtf = subtf
        self.stops = stops
        self.vocab = vocab
        self.encoding = encoding
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.docsize = docsize


    def load_data(self):
        logging.info('载入数据from：%s' % str(self.home_data+self.input_df))
        weibo = pd.read_csv(self.home_data+self.input_df, sep=',', encoding='utf-8')
        weibo.columns = self.colname
        weibo['用户ID'] = weibo['用户ID'].fillna('')
        weibo['aite'] = weibo['aite'].fillna('')
        weibo = weibo[weibo['aite']!='']        # 剩63708个用户
        self.weibo = weibo
        users = list(weibo['用户ID'])
        self.users = users
        mentions = list(weibo['aite'])
        self.mentions = mentions


        # 合并每个用户发布的所有的微博

    def get_graph(self):
        g = nx.Graph()
        nodes = self.users
        node_id = {node: ID for ID, node in enumerate(nodes)}
        with open('node_id.pkl', 'wb') as f:
            pickle.dump(node_id, f)
        self.node_id = node_id
        g.add_nodes_from(list(node_id.values()))       # 用户ID对应的id数字作为节点
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])  # 添加自环

        # 添加所有的边
        logging.info('添加节点成功，开始添加所有的边')
        for i in range(len(self.users)):
            user = nodes[i]                 # 第i个用户的用户ID
            user_id = node_id[user]         # user_id = 第i个用户的ID
            mention = self.mentions[i]      # 第i个用户@的对象
            mention = mention.split(',')    # 第i个用户@对象列表
            mention = list(set(mention))    # 第i个用户@ 的对象用户ID
            idmentions = set()
            for m in mention:               # 第i个用户@的每一个对象
                if m in node_id:            # 如果这个@对象，也在用户集里面
                    idmentions.add(node_id[m])  # idmentions添加这个@对象的ID
                else:
                    ID = len(node_id)       # 如果这个@对象不在用户集里面。
                    node_id[m] = ID         # 在node_id中添加这个用户
                    idmentions.add(ID)      # idmentions添加这个@对象的ID
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)  # 把这些被@到的用户也添加到图的节点
            for ID in idmentions:              # 第i个用户@的所有对象id
                g.add_edge(ID, user_id)        # 添加@对象和第i个用户的边
        self.g = g

        celebrities = []  # 名人集
        for i in range(len(nodes), len(node_id)):  # nodes_list9000多个用户，node_id9000+@用户
            deg = len(g[i])  # 被@用户的邻居信息
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)  # 如果这个被@的用户是孤立的或者邻居的数量大于10
        logging.info(
            'removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)  # 移除名人节点，以及所有的边
        logging.info('移除名人节点后的图g #nodes: %d, #edges: %d' % (nx.number_of_nodes(g), nx.number_of_edges(g)))
        self.biggraph = g  # 中间的图保存。
        logging.info('projecting the graph')
        projected_g = efficient_collaboration_weighted_projected_graph2(g, list(range(len(nodes))))  # 最终图。
        logging.info('最终图 #nodes: %d, #edges: %d' % (nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        self.graph = projected_g  # self.graph变量被更新为最终图
        nx.write_adjlist(self.graph, self.home_data+'graph.dat')
        logging.info('Graph保存为: %s' % str(self.home_data+'graph.dat'))

    def delete_guli_nodes(self):
        weibo = self.weibo
        users = self.users
        graph = self.graph
        node_id = self.node_id
        guli_nodes = []
        for ele in list(graph.nodes()):
            nei = list(graph.neighbors(ele))
            if len(nei) == 0:
                guli_nodes.append(ele)
        graph.remove_nodes_from(guli_nodes)
        self.graph = graph      # 删除了3w多个节点,剩余31147个节点，306925条边
        logging.info('最终图 #nodes: %d, #edges: %d' % (nx.number_of_nodes(graph), nx.number_of_edges(graph)))
        shengyu_node = list(graph.nodes())
        shengyu_node = [int(ele) for ele in shengyu_node]
        shengyu_user = []
        for k, v in node_id.items():
            if v in shengyu_node:
                shengyu_user.append(k)
        weibo = weibo[weibo['用户ID'].isin(shengyu_user)]
        self.weibo = weibo
        self.users = list(weibo['用户ID'])


    def tfidf(self):
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                    norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                    min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1), stop_words=self.stops,
                                     vocabulary=self.vocab, encoding=self.encoding, dtype='float32')
        logging.info(self.vectorizer)
        self.X = self.vectorizer.fit_transform(self.weibo.content.values)
        logging.info("tfidf   n_samples: %d, n_features: %d" % self.X.shape)  # (5685, 9204)

    def doc2vec(self):
        d2v_model = os.path.join(self.home_data, 'dbow.model')
        if os.path.exists(d2v_model):
            print("直接读取已经存储的doc2vec模型")
            model_dbow = gensim.models.Doc2Vec.load(self.home_data + "/dbow.model")
        else:
            print("没有doc2vec模型，需要重新训练")
            text = self.weibo.content.values
            text = [sen.split() for sen in text]
            text = labelizeContent(text, 'train')
            data = text[:]
            model_dbow = Doc2Vec(dm=0, min_count=1, window=10, size=self.docsize, sample=1e-3, negative=5, workers=3)
            model_dbow.build_vocab(data[:])

            for epoch in range(20):
                all_reviews = data[:]
                random.shuffle(all_reviews)
                t_epoch = time.time()
                model_dbow.train(text, total_examples=model_dbow.corpus_count, epochs=3)  # epochs 设置为 1
                model_dbow.alpha -= 0.002
                model_dbow.min_alpha = model_dbow.alpha
                print('=' * 30 + str(epoch) + '训练模型{:.4f}mins'.format((time.time() - t_epoch) / 60) + '=' * 30)
                model_dbow.save(self.home_data+"/dbow.model")
        pdb.set_trace()
        # d2v = get_d2v(model_dbow, self.users)
        d2v = getVecs(model_dbow, data)
        pdb.set_trace()
        self.d2v = d2v


    def assignlabel(self):
        labels = list(self.weibo.label)
        label_map = {j: i for i, j in enumerate(list(set(labels)))}
        labels = np.array(list(map(label_map.get, labels)))
        self.Y = labels
        logging.info("标签类别Y: %s" % str(self.Y.shape))


if __name__ == '__main__':
    home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"
    input_df = "16w_user_340wweibo_merge_province_label164016.csv"
    colname = ['content', 'label', '用户ID', '用户名', 'aite', 'content_all']   # label = province
    input_df = "837908uid_province_content_aite.csv"
    colname = ['aite', 'content', 'label', '用户ID']

    data = Graph_class(home_data=home_data, input_df=input_df, colname=colname,
                       celebrity_threshold=15, docsize=5000, mindf=3)
    data.load_data()
    print("载入数据成功，开始建模图")
    data.get_graph()
    print("图建模成功，#nodes: %d, #edges: %d" % (nx.number_of_nodes(data.graph), nx.number_of_edges(data.graph)))
    data.delete_guli_nodes()
    print("图建模成功，#nodes: %d, #edges: %d" % (nx.number_of_nodes(data.graph), nx.number_of_edges(data.graph)))
    data.tfidf()
    print("tfidf向量转换成功", data.X.shape)
    # data.doc2vec()
    # print("doc2vec向量转换成功", data.d2v.shape)
    data.assignlabel()
    nx.write_gexf(data.graph, 'graph.gexf')
    pdb.set_trace()