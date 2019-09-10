# -*- coding:utf-8 -*-
__author__ = 'GIS'
'''
1， 读取带有标签df1和微博内容df2的数据。
2， 以用户为单位将，用户的标签和微博数据合并，将用户发布的所有的微博数据合并为content_all
3,  合并用户的@对象和转发对象。
'''
import pandas as pd
import re




def extract_aite(df):  # 提取@XX,添加一列aite, 删除了被转发的@ 对象
    pattern = "[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?"
    wbnr = list(df['微博内容'])
    wb = []
    for sentence in wbnr:
        sentence = re.sub('//@.*?:', ' ', sentence)   # 将转发的@删除
        wb.append(sentence)
    ob = [re.sub("[\[\]\@\'\ ]", '', str(re.findall(pattern, str(w)))).strip() for w in wb]
    df['aite'] = ob
    return df



class Ready_Graph():
    def __init__(self, home_data, input_df1, input_df2, colname1, colname2, encoding='utf-8'):
        self.home_data = home_data
        self.input_df1 = input_df1
        self.input_df2 = input_df2
        self.colname1 = colname1
        self.colname2 = colname2
        self.encoding = encoding

    def read_data(self):
        df1 = pd.read_csv(self.home_data+self.input_df1,
                          sep=',', encoding='utf-8')
        df1.columns = self.colname1
        self.df1 = df1

        df2 = pd.read_csv(self.home_data+self.input_df2,
                          sep=',', encoding='utf-8',
                          usecols=['用户ID', '微博内容'])     # 在df2中选用用户ID、 微博内容两个字段。
        df1.columns = self.colname2
        self.df2 = df2

    def merge_df(self):
        assert self.df1.shape[0] == self.df2.shape[0], "输入df1与输入df2的行数不相等"
        df2 = extract_aite(self.df2)     # 添加aite列，转发的@对象被删除。
