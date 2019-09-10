# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:28:57 2019

@author: GIS
"""

import pandas as pd
import re

df = pd.read_csv("E:\\deep_network\\GCN Highway My Data\\data\\weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol_userlabel837908.csv", sep=',',
                 encoding='utf-8')
df = df.drop_duplicates(['用户ID','content'])
test = df.iloc[0:100,:]
user_id = pd.read_csv("E:\\deep_network\\GCN Highway My Data\\data\\weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol.csv", sep=',',
                 encoding='utf-8', usecols=['用户ID','微博内容'])
yonghuid = list(user_id['用户ID'])
yonghuming = list(user_id['用户名'])
users = list(df['用户ID'])
df['用户名'] = yonghuming


df = df.drop_duplicates(subset=['content', 'label', '用户ID'])
users = list(df['用户ID'])

weibo = pd.read_csv("E:\\deep_network\\GCN Highway My Data\\data\\weiboallusers_locate_210000nerusers_data_process.csv", sep=',',
                 encoding='utf-8', usecols=['用户ID', 'aite', 'jieba_stopword'])
weibo['aite'] = weibo['aite'].fillna('')
weibo['jieba_stopword'] = weibo['jieba_stopword'].fillna('')
nodes = list(set(list(weibo['用户ID'])))

df_weibo = df[df['用户ID'].isin(nodes)]
nodes = list(df_weibo['用户ID'])

# 提取16万用户的艾特对象和所有的微博内容
aite_list = []
content_list = []
for i, user in enumerate(nodes):
    data = weibo[weibo['用户ID']==user]
    aite = list(data['aite'])
    aite = ','.join(aite)
    aite_list.append(aite)
    
    content = list(data['jieba_stopword'])
    content = ' '.join(content)
    content_list.append(content)
    if i % 10000 == 0:
        print(i, aite)
        


aite_list1 = [ele.strip(',').split(',') for ele in aite_list]

aite_list2 = []
for i, line in enumerate(aite_list1):
    print(i)
    linshi = []
    for ele in line:
        if ele == '':
            pass
        else:
            linshi.append(ele)
    aite_list2.append(linshi)
aite_list3 = [','.join(ele) for ele in aite_list2]












df_weibo['aite'] = aite_list3
df_weibo['content_all'] = content_list

df_weibo.to_csv("E:\\deep_network\\GCN Highway My Data\\data\\16w用户340w微博合并+省标签164016.csv",
                sep=',', encoding='utf-8', index=False)
df_weibo.shape
df_weibo.columns

