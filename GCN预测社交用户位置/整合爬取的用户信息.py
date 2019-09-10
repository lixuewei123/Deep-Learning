# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import numpy as np


path = "E:\\deep_network\\GCN Highway My Data\\data"
users1 = pd.read_csv(path+'/users55w.csv', encoding='utf-8', header=None)

users2 = pd.read_csv(path+'/users_94623aite.csv', encoding='gb18030', header=None)

users = users1
users = users.append(users2)
users.to_csv(path+'649550users.csv', index=False, header=False, encoding='utf-8')

