# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import numpy as np

path = "E:\\deep_network\\GCN Highway My Data\\data"
users = pd.read_csv(path+'/data649550users.csv', encoding='utf-8', header=None)

weibo =  pd.read_csv(path+'\\16w_user_340wweibo_merge_province_label164016.csv',
                     encoding='utf-8')
weibo.aite = weibo.aite.fillna('')
aite = list(weibo.aite)

users.iloc[0, :]

users3 = users[users[8]!='no']  # 516838
users3 = users3
name = list(users[1])
loc = list(users[8])
usertype = list(users[2])

name_type = {na : ty for na, ty in zip(name, usertype)}

aite1[1]
aite1 = []
for ele in aite:
    ele = ele.split(',')
    aite1.append(ele)

aite2 = []
for i, ele in enumerate(aite1):
    if i % 1000 == 0:
        print(i)
    liebiao = list(map(name_type.get, ele))
    ele1 = []
    for user, renzhen in zip(ele, liebiao):
        if renzhen != '微博个人认证' and renzhen != '微博官方认证':
            ele1.append(user)
    aite2.append(ele1)


weibo['aite'] = aite2
weibo.to_csv(path+'\\16w_user_340wweibo_merge_province_label164016_delaite1.csv',
             encoding='utf-8', index=False)