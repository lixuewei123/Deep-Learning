'''
1， 读取带有标签df1和微博内容df2的数据。
2， 以用户为单位将，用户的标签和微博数据合并，将用户发布的所有的微博数据合并为content_all
3,  合并用户的@对象和转发对象。
'''
import pandas as pd
import re
import pickle


home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"
input_df1 = "weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol_userlabel837908.csv"
input_df2 = "weibo_locate_1758707_jieba-delete_stopword_jianti_labeled_1294347_ner_delcol.csv"
colname1 = ['city', 'content', 'province', '用户ID', '用户名']
colname2 = ['用户ID', '微博内容']
colname3 = ['是否转发', '用户ID', '用户名', 
               '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞',
               '发文设备', '签到地点', '发文时间', 
               '被转发用户名', '被转发用户类型', '被转发微博内容',
               '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数']


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


def read_data(home_data, input_df1, input_df2, colname1, colname2):
    df1 = pd.read_csv(home_data+input_df1,
                          sep=',', encoding='utf-8')
    df1.columns = colname1

    df2 = pd.read_csv(home_data+input_df2,
                          sep=',', encoding='utf-8',
                          usecols=['用户ID', '微博内容'])     # 在df2中选用用户ID、 微博内容两个字段。
    df2.columns = colname2
    return df1, df2


# ====================== 读取数据 =============================================
df1, df2 = read_data(home_data, input_df1, input_df2, colname1, colname2)

t1 = df2.iloc[0:100, :]
t1 = list(df1['用户ID'])
t1 = list(set(t1))
with open(home_data+"users.pkl", 'wb') as f:
    pickle.dump(t1, f)
t2 = list(df2['用户ID'])      # df1   df2两个用户ID的排列顺序是一样的
df2 = extract_aite(df2)

# ===================== 合并数据 ==============================================
users = list(set(list(df2['用户ID'])))
df1 = df1.drop_duplicates(['用户ID'])
df2 = df2.drop_duplicates(['用户ID'])

# ==================== 所有的微博内容 ==========================================
df3 = pd.read_csv("E:\\deep_network\\GCN Highway My Data\\data\\weiboallusers_locate_837908users_delcol.csv", 
                  encoding='utf-8', chunksize=1000000, header=None)
for i, chunk in enumerate(df3):
    chunk.columns = colname3
    chunk = chunk[chunk['用户ID'].isin(users)]
    chunk = chunk.drop_duplicates(['用户ID', '微博内容'])
    print(i, chunk.shape)
    chunk.to_csv("E:\\deep_network\\GCN Highway My Data\\data\\weiboallusers_locate_837908users_delcol_dropduplicates.csv",
                 sep=',', header=False, index=False, encoding='utf-8',
                 mode='a')

df3 = pd.read_csv("E:\\deep_network\\GCN Highway My Data\\data\\weiboallusers_locate_837908users_delcol_dropduplicates.csv", 
                  encoding='utf-8', usecols=[1, 5, 14], header=None) 
df3.columns = ['用户ID', '微博内容', '被转发微博内容']
df3 = df3.drop_duplicates(['用户ID', '微博内容'])     # 舍弃转发内容中@的对象。4619052条微博。
df3 = extract_aite(df3)
df3['被转发微博内容'] = df3['被转发微博内容'].fillna('')
df3['内容'] = df3['微博内容'] + df3['被转发微博内容'] 
test = df3.iloc[0:100, :]


# =================== 合并label和aite =========================================
users = list(df1['用户ID'])

import time
t = time.time()
content = []
aite = []
for i, user in enumerate(users):
    if i % 10000 == 0:
        print(i, (time.time()-t)/60, 'mins')
    ceshi = df3[df3['用户ID'] == user]
    aite_ele = list(ceshi['aite'])
    aite_ele = ','.join(aite_ele).strip(',')
    content_ele = list(ceshi['内容'])
    content_ele = ' '.join(content_ele)
    aite.append(aite_ele)
    content.append(content_ele)

df = df1
df['aite'] = aite
df['content_all'] = content
df.to_csv("E:\\deep_network\\GCN Highway My Data\\data\\weiboallusers_locate_837908users_delcol_dropduplicates_labeled_aite_contentall.csv",
                 sep=',', index=False, encoding='utf-8')