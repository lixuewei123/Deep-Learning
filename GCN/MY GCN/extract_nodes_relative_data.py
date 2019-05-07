import numpy as np
import pandas as pd
import re


def load_data(nodes_list):
    paths = []
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong0.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong1.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong2.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong3.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong4.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong5.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong6.csv")
    paths.append("E:/lxw_data/data/regulation data/df_aite_user_putong7.csv")
    colname = ['是否转发', '用户ID', '用户名',
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞', '微博URL',
               '发文设备', '签到地点', '发文时间', '被转发用户ID',
               '被转发用户名', '被转发用户类型', '被转发微博ID', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
               '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', 'aite']
    for j, path in enumerate(paths):
        print(path)
        df = pd.read_csv(path, sep=',',
                         encoding='utf-8', header=None, chunksize=100000, error_bad_lines=False)
        for i, chunk in enumerate(df):
            print(i, chunk.shape)
            chunk.columns = colname
            chunk = chunk[chunk['用户名'].isin(nodes_list) & chunk['aite'].isin(nodes_list)]
            print(i, chunk.shape)
            chunk.to_csv('E:/lxw_data/data/regulation data/idea1/mutual_aite_yu' + str(j) + '.csv', sep=',',
                         header=False, index=False, mode='a')


def load_node(path):
    with open(path, 'r') as f:
        nodes_list3 = f.readlines()
    nodes_list = []
    for node in nodes_list3:
        node = re.sub('\n', '', node)
        nodes_list.append(node)
    return nodes_list


def main():
    nodes_list = load_node("E:/lxw_data/data/regulation data/idea1/nodes_list.txt")
    load_data(nodes_list)


if __name__ == '__main__':
    main()