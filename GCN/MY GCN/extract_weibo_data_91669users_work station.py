import numpy as np
import pandas as pd
import re

def search_data(nodes):
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
            chunk = chunk[['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间']]
            chunk = chunk[chunk['用户名'].isin(nodes)]
            print(i, chunk.shape)
            chunk.to_csv('E:/lxw_data/data/regulation data/users_text'+str(j)+'.csv', sep=',', header=False, index=False, mode='a')


def load_node(path):
    with open(path, 'r') as f:
        nodes_list3 = f.readlines()
    nodes_list = []
    for node in nodes_list3:
        node = re.sub('\n', '', node)
        nodes_list.append(node)
    return nodes_list


def main():
    nodes = load_node("E:/lxw_data/data/regulation data/idea1/nodes10w.txt")
    print(len(nodes))
    search_data(nodes)


if __name__ == '__main__':
    main()