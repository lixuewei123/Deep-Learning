# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import re
import numpy as np

colname = ['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', 'aite', 'del', 'url']


def read_all_data():
    df = pd.read_csv("E:/deep_network/deal_data/regulation data/aite_data/relative_weibo.csv",
                     sep=',', header=None, encoding='utf-8')
    df.columns = colname
    return df


def read_nodes():
    with open("E:/deep_network/deal_data/regulation data/nodes10w.txt", 'r') as f:
        nodes_list1 = f.readlines()
    nodes_list2 = []
    for node in nodes_list1:
        node = re.sub('\n', '', node)
        nodes_list2.append(node)
    return nodes_list2


def extra_edges(df, nodes):
    start_users = list(df['用户名'])
    end_users1 = list(df['aite'])
    end_users2 = []
    for users in end_users1:
        users = users.split(',')
        end_users2.append(users)
    end_users = end_users2

    # 直接相互@ 的边
    edges = []
    for i in range(len(start_users)):
        for ele in end_users[i]:
            edges.append((start_users[i], ele))
    print(len(edges))
    edges1 = []
    for i, edge in enumerate(edges):
        if i % 1000 == 0:
            print('边', i)
        if edge[0] in nodes or edge[1] in nodes:
            edges1.append(edge)
    print(len(edges1))
    # 共同@同一对象的边
    return edges1


def main():
    df = read_all_data()
    nodes = read_nodes()
    edges = extra_edges(df, nodes)
    # 写出边
    with open("E:/deep_network/deal_data/regulation data/edges180w_aite .txt", 'w+') as f:
        for edge in edges:
            try:
                f.write(str(edge)+'\n')
            except:
                print(edge)


if __name__ == '__main__':
    main()