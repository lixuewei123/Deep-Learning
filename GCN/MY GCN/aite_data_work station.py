
import numpy as np
import pandas as pd
import re

def read_data():
    colname = ['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
                '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', 'aite']
    df = pd.read_csv("E:/lxw_data/data/regulation data/loc-onlyaite.csv", sep=',', header=None)  # 2830171条微博
    df.columns = colname
    return df


def ready_data(df):
    aite_object = list(df['aite'])
    aite_object1 = []
    for ele in aite_object:
        ele = ele.split(',')
        for e in ele:
            aite_object1.append(e)
    aite_object1 = list(set(aite_object1))  # 515948个@的用户
    return aite_object1


def extract_aite(df):  # 提取@XX,添加一列aite
    pattern = "[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?"
    wbnr = list(df['微博内容'])
    ob = [re.sub("[\[\]\@\'\ ]", '', str(re.findall(pattern, str(w)))).strip() for w in wbnr]
    df['aite'] = ob
    return df


def del_aite(df):
    wbnr1 = list(df['微博内容'])
    pattern1 = "//@"
    ob1 = []
    for w in wbnr1:
        if pattern in str(w):
            ob1.append(0)
        else:
            ob1.append(1)
    df['del'] = ob1
    df = df[df['del']==1]
    return df


def connect_url(df):
    user = list(df['用户ID'])
    weibo = list(df['微博URL'])
    # urls = []
    # for u, w in zip(user, weibo):
    #     url = "https://weibo.com/" + str(u) + "/"+ str(w) + "?type=comment"
    #     urls.append(url)
    urls = ["https://weibo.com/" + str(u) + "/"+ str(w) + "?type=comment" for u, w in zip(user, weibo)]
    df['url'] = urls
    return df


def search_data():
    # users = []
    # type = []
    path = []
    path.append("E:/weibo_data/weibo_freshdata.2018-09-30")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-01")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-02")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-03")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-04")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-05")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-06")
    path.append("E:/weibo_data/weibo_freshdata.2018-10-07")
    colname = ['ID', '抓取时间', '抓取时间戳', '是否转发', '用户ID', '用户名', '头像URL',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '发文时间戳', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间',
               '被转发微博的时间戳', '微博配图', '', '', 'video', 'video-image-URL', '', '', '', '', '日期']

    for j, pa in enumerate(path):
        print(pa)
        df = pd.read_csv(pa, sep='\t',
                         encoding='utf-8', header=None, chunksize=1000000, error_bad_lines=False)
        for i, chunk in enumerate(df):
            print(i, chunk.shape)
            chunk.columns = colname
            # chunk.drop_duplicates(subset='被转发用户名', inplace=True)
            # print(chunk.shape)
            # chunk['被转发用户名'] = chunk['被转发用户名'].fillna(0)
            # chunk = chunk[chunk['被转发用户名'].isin(aite)]
            # users.extend(list(chunk['被转发用户名']))
            # type.extend(list(chunk['被转发用户类型']))
            # 提取艾特对象数据
            chunk = chunk[['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间']]
            chunk = extract_aite(chunk)
            chunk = del_aite(chunk)
            chunk = connect_url(chunk)
            chunk['aite'] = chunk['aite'].fillna(0)
            chunk = chunk[chunk['aite']!=0]
            chunk = chunk[chunk['aite']!='']
            chunk = chunk[chunk['用户类型']=='普通用户']
            print(i, chunk.shape)
            chunk.to_csv('E:/lxw_data/data/regulation data/df_aite_user_putong'+str(j)+'.csv', sep=',', header=False, index=False, mode='a')




#
# def main():
#     df = read_data()
#     aite = ready_data(df)
#     print('@对象有'+str(len(aite)))
#     users_list, type_list = search_data(aite)
#     data = pd.DataFrame()
#     data['user'] = users_list
#     data['type'] = type_list
#     data.drop_duplicates(subset='user', inplace=True)
#     data.to_csv("E:/lxw_data/data/regulation data/aite_type1.csv", sep=',', header=False, index=False,mode='a')


def main():
    search_data()



if __name__ == '__main__':
    main()