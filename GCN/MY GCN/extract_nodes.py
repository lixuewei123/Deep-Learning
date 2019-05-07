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

paths = []
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong0.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong1.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong2.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong3.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong4.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong5.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong6.csv")
paths.append("E:/deep_network/deal_data/regulation data/aite_data/df_aite_user_putong7.csv")


def read_crawal_user(path):
    df = pd.read_csv(path, sep=',', header=None, encoding='utf=8')

    df.columns = ['ID', 'name', 'type', 'url', 'guanzhu', 'fensi', 'weibo', 'xingbie',
                  'address', 'jianjie', 'biaoqian', 'jiaoyu']
    df = df.drop_duplicates(subset='name')
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

    print('去微博前', df.shape)
    weibo = list(df['weibo'])
    pattern = '万'
    weibo2 = []
    weibo3 = []
    for wei in weibo:
        if pattern in wei:
            weibo2.append(0)
            weibo3.append(0)
        else:
            weibo2.append(1)
            weibo3.append(int(wei))
    df['del1'] = weibo2
    df['del2'] = weibo3
    df = df[df['del1'] != 0]
    df = df[df['del2'] != 0]
    df = df[df['del2'] <= 10000]
    print('去微博后', df.shape)

    guanzhu = list(df['fensi'])
    # print('粉丝', len(guanzhu))
    g2 = []
    g3 = []
    for g in guanzhu:
        if pattern in g:
            g2.append(0)
            g3.append(0)
        else:
            g2.append(1)
            g3.append(int(g))
    df['del3'] = g2
    df['del4'] = g3
    df = df[df['del3'] != 0]
    df = df[df['del4'] != 0]
    df = df[df['del4'] <= 1000]
    print('去粉丝后', df.shape)

    df = df.reset_index(drop=True)
    return df


def extra_cross_user(nodes2):  # 返回用于建立图的节点用户
    with open("E:/deep_network/deal_data/regulation data/nodes.txt", 'r') as f:
        nodes_list1 = f.readlines()
    nodes_list2 = []
    for node in nodes_list1:
        node = re.sub('\n', '', node)
        nodes_list2.append(node)
    nodes3 = list(set(nodes2).intersection(set(nodes_list2)))
    return nodes3


def extra_weibo(paths, users):
    for path in paths:
        print(path)
        da1 = pd.read_csv(path, sep=',', header=None, encoding='utf-8', chunksize=100000)
        for i, chunk in enumerate(da1):
            print(i, chunk.shape)
            chunk.columns = colname
            chunk = chunk[chunk['用户名'].isin(users) | chunk['aite'].isin(users)]
            print(i, chunk.shape)
            chunk.to_csv("E:/deep_network/deal_data/regulation data/aite_data/relative_weibo.csv",
                         sep=',', encoding='utf-8', header=None, index=False, mode='a')


def main():
    crawal_user = read_crawal_user("E:/deep_network/deal_data/regulation data/LXW/users40w.csv")  # 抓取的所有用户
    nodes = list(crawal_user['name'])
    users = extra_cross_user(nodes)
    with open("E:/deep_network/deal_data/regulation data/nodes10w.txt", 'w+') as f:
        for node in users:
            f.write(node+'\n')
    print('所有用户数量：', len(users))
    # 输出节点用户属性
    crawal_user = crawal_user[crawal_user['name'].isin(users)]
    print(crawal_user.shape)
    crawal_user.to_csv("E:/deep_network/deal_data/regulation data/10w_users.csv",
                         sep=',', encoding='utf-8', header=None, index=False, mode='a')
    # 提取相关的微博
    # extra_weibo(paths, users)


if __name__ == '__main__':
    main()