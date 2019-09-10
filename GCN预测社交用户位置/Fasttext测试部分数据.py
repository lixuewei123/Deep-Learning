# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:47:41 2019

@author: GIS
"""

import networkx as nx


G = nx.read_adjlist('E:\\deep_network\\GCN Highway My Data\\data\\graph.dat')
G.number_of_nodes()
G.number_of_edges()


for node in list(G.nodes()):
    nei = list(G.neighbors(node))
    if len(nei) == 0:
        G.remove_node(node)
        
        
import pickle
with open('E:\\deep_network\\GCN Highway My Data\\data\\node_id.pkl', 'rb') as f:
    node_id = pickle.load(f)

nodes = list(G.nodes())

names = []
for i, node in enumerate(nodes):
    print(i)
    for k, v in node_id.items():
        if int(node) == v:
            names.append(k)

import pandas as pd
weibo = pd.read_csv('E:\\deep_network\\GCN Highway My Data\\data\\16w_user_340wweibo_merge_province_label164016_delaite.csv',
                    encoding='utf-8')
aite = list(weibo.aite)

aite = [ele.replace('[','').replace(']','').replace("'", '') for ele in aite]
weibo.aite = weibo.aite.astype(str)
weibo.aite = aite

weibo1 = weibo[weibo.aite != '']



weibo1 = weibo[weibo.用户名.isin(names)]
weibo1.to_csv('E:\\deep_network\\GCN Highway My Data\\data\\16w_user_340wweibo_merge_province_label164016_delaite_25276.csv',
              index=False, encoding='utf-8')