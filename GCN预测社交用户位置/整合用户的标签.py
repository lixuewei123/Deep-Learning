# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import collections
from collections import Counter


class add_users_address():
    def __init__(self, home_data, input_df, output_df, colname, label_column, content_column):
        self.home_data = home_data
        self.input_df = input_df
        self.output_df = output_df
        self.colname = colname
        self.label_column = label_column    # 默认用province
        self.content_column = content_column

    def read_data(self):
        weibo = pd.read_csv(self.home_data+self.input_df, sep=',', encoding='utf-8')
        weibo.columns = self.colname
        self.weibo = weibo

        users = list(weibo['用户名'])
        users_content = []
        users_address = []
        users_city = []
        users_id = []
        for i, user in enumerate(users):
            df = weibo[weibo['用户名']==user]
            content = list(df[self.content_column])
            content = ' '.join(content)             # 合并了用户发布的微博内容
            users_content.append(content)
            # 用户的艾特对象
            adds = list(df[self.label_column])
            adds = ','.join(adds)
            users_address.append(adds)

            adds_city = list(df['city'])
            adds_city = ','.join(adds_city)
            users_city.append(adds_city)
            # 用户ID
            idid = list(df['用户ID'])[0]
            users_id.append(idid)

            if i % 10000 == 0:
                print(i)

        self.users = users
        self.users_content = users_content
        self.users_address = users_address
        self.users_city = users_city
        self.users_id = users_id

    def get_label(self):
        # rule1: 去重只有一个位置的，自然选择这一个
        # rule2: 去重有多个位置的，选择最多的哪个位置
        # rule3: 位置数量一样多的，选择第一个
        users = self.users
        users_address = self.users_address
        users_address_filter = []
        for i, adds in enumerate(users_address):
            adds = adds.split(',')
            adds_store = adds
            adds = list(set(adds))
            if len(adds) == 1:
                users_address_filter.append(adds[0])
            else:
                count = Counter(adds_store)
                t = list(count.values())
                pre_add = []
                for k, v in count.items():
                    if v == max(t):
                        pre_add.append(k)
                # 添加最多的地址中的第一个地址
                users_address_filter.append(pre_add[0])
        assert len(users_address_filter) == len(users), "地址数量与用户数量不相等"
        self.users_address = users_address_filter
        c = {'用户ID': self.users_id, '用户名': self.users, 'province': self.users_address, 'content': self.users_content, 'city': self.users_city}
        self.user_add_df = pd.DataFrame(c)

        self.user_add_df.to_csv(self.home_data+self.output_df, sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"
    colname = ['发文时间', '微博ID', '用户ID', '用户名', 'URL', '纬度', '经度', '签到', '日期', '是否转发',
       '用户类型', '微博内容', '转发', '评论', '点赞', '发文设备', 'wbnr_clean', 'segment_jieba',
       'aite', 'aitelist', 'label', 'jianti', 'segment_pku', 'stopword',
       'province', 'city', 'district', 'street', 'street_number', 'business',
       'cityCode', '签到城市', 'Id', '发文时间.1', 'hanlp', 'ner']

    label_columns = 'province'
    label_columns1 = 'city'
    content_column = 'stopword'

    data = add_users_address(home_data=home_data,
                             input_df="weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol.csv",
                             output_df="weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol_userlabel837908.csv",
                             colname=colname, label_column=label_columns, content_column=content_column)

    data.read_data()
    data.get_label()
