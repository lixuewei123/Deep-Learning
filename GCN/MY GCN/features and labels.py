import pandas as pd
import numpy as np
import re


users_df = pd.read_csv("E:\\lxw_data\\GCN\\data\\users40w.csv", header=None)
users_df.columns = ['id', 'name', 'type', 'url', 'guanzhu', 'fensi', 'weibo', 'xingbie',
                    'loc', 'biaoqian', 'jianjie', 'jiaoyu']

def add_label(df):
    loc = df['loc']
    locs = []
    for l in loc:
        l = l[0:2]
        locs.append(l)
    df['label'] = locs
    return df


def read_nodes():
    with open("E:/lxw_data/GCN/data/nodes10w.txt", 'r') as f:
        nodes_list1 = f.readlines()
    nodes_list2 = []
    for node in nodes_list1:
        node = re.sub('\n', '', node)
        nodes_list2.append(node)
    return nodes_list2


def read_edges():
    with open("E:/lxw_data/GCN/data/edges36w .txt", 'r') as f:
        edges_list1 = f.readlines()

    i = 0
    edges = []
    for ed in edges_list1:
        i += 1
        # print(i)
        ed = re.sub("[\(\)\'\ ]", '', ed).strip()
        ed = ed.split(',')
        edges.append(ed)

    edges_df = pd.DataFrame(edges)
    return edges_df


def edges_dataframe(nodes, edges_df):
    # =================================================================
    #                       删除孤立节点
    # =================================================================
    s1 = list(edges_df[0])
    s2 = list(edges_df[1])
    s1.extend(s2)
    s1 = list(set(s1))  # 去重复
    nodes = s1  # 更新节点
    # =================================================================
    node_map = {j: i for i, j in enumerate(nodes)}
    e1 = edges_df[0]
    e2 = edges_df[1]
    e11 = list(map(node_map.get, e1))
    e22 = list(map(node_map.get, e2))
    edges_df[2] = e11
    edges_df[3] = e22
    edges_df = edges_df[edges_df[2]!=edges_df[3]]
    return edges_df, nodes

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


users_df = add_label(users_df)

nodes = read_nodes()
edges_df = read_edges()
edges_df, nodes = edges_dataframe(nodes, edges_df)

print('获取节点的标签')
labels = []
for i, node in enumerate(nodes):
    print(i)
    ss = users_df[users_df['name']==node]
    s = list(ss['label'])[0]
    labels.append(s)

features = encode_onehot(labels)
print('features.shape', features.shape)
np.save("E:/lxw_data/GCN/data/features.npy", features)
label = np.where(features)[1]
print('label.shape', label.shape)
np.save("E:/lxw_data/GCN/data/label.npy", label)