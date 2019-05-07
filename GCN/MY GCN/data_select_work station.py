
import re
import os
import numpy as np
import pandas as pd

def read_data(path):
    df = pd.DataFrame()
    for i, pa in enumerate(os.listdir(path)):
        pa = path + '/' + pa
        print(i,pa)
        df_linshi = pd.read_csv(pa, sep=',', header=None)
        df = pd.concat([df, df_linshi])
    return df

def read_data2(path):  # 读取数据，筛选列
    colname = ['索引','ID', '是否转发', '用户ID', '用户名', '头像URL',
           '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
           '发文设备', '签到地点', '发文时间', '发文时间戳', '被转发用户ID',
           '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
           '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
           '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', '被转发微博的时间戳']
    df = pd.read_csv(path, sep=',', header=None)
    df.columns = colname
    df = df[['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间']]
    return df


def extract_aite(df):  # 提取@XX,添加一列aite
    pattern = "[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?"
    wbnr = list(df['微博内容'])
    ob = [re.sub("[\[\]\@\'\ ]", '', str(re.findall(pattern, str(w)))).strip() for w in wbnr]
    df['aite'] = ob
    return df






def main1():
    path = "E:/lxw_data/data/loc_relative_data"
    pa = []
    pa.append('loc-relative-2018-09-30.csv')
    pa.append('loc-relative-2018-10-01.csv')
    pa.append('loc-relative-2018-10-02.csv')
    pa.append('loc-relative-2018-10-03.csv')
    pa.append('loc-relative-2018-10-04.csv')
    pa.append('loc-relative-2018-10-05.csv')
    pa.append('loc-relative-2018-10-06.csv')
    pa.append('loc-relative-2018-10-07.csv')
    for pat in pa:
        print(path+'/'+pat)
        df = read_data2(path+'/'+pat)
        df = extract_aite(df)
        df.to_csv("E:/lxw_data/data/regulation data/loc-aite.csv", sep=',', header=False, index=False,mode='a')


def main():
    colname = ['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', 'aite']
    df = pd.read_csv("E:/lxw_data/data/regulation data/loc-aite.csv", sep=',',
                     encoding='utf-8', header=None, chunksize=1000000, error_bad_lines=False)
    for i, chunk in enumerate(df):
        print(i, chunk.shape)
        chunk.columns = colname
        chunk['aite'] = chunk['aite'].fillna(0)
        chunk = chunk[chunk['aite']!=0]
        chunk.to_csv("E:/lxw_data/data/regulation data/loc-onlyaite.csv", sep=',', header=False, index=False,mode='a')


if __name__ == '__main__':
    main()



