# -*- coding:utf-8 -*-
__author__ = 'GIS'

import fastText.FastText as ft
import pandas as pd
import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO,
                    filename="E:/deep_network/GCN Highway My Data/fasttext.log",
                    filemode='w+')


class Fasttext_test():
    def __init__(self, home_data, input_df, colname):
        self.home_data = home_data
        self.input_df = input_df
        self.colname = colname

    def load_data(self):
        model_file = os.path.join(self.home_data, 'fasttext.model')
        if os.path.exists(model_file):
            pass
        else:
            logging.info('loading the dataset from %s' % self.home_data)
            logging.info('loading the dataset from %s' % self.input_df)
            weibo = pd.read_csv(self.home_data+self.input_df, sep=',', encoding='utf-8')
            weibo.columns = self.colname
            # weibo['ner'] = weibo['ner'].fillna('')
            # weibo = weibo[weibo['ner']!='']
            assert 'jieba_stopword' in self.colname, "jieba_stopword字段不存在 %s"% str(self.colname)
            assert 'province' in self.colname, "province字段不存在 %s" % str(self.colname)
            weibo['jieba_stopword'] = weibo['jieba_stopword'].fillna('')
            weibo = weibo[weibo['jieba_stopword'] != '']
            wbnr = list(weibo['jieba_stopword'])
            wbnr = [line.split(' ') for line in wbnr]
            del_index = []
            for i, line in enumerate(wbnr):
                if len(line) < 3:
                    del_index.append(i)
            weibo.reset_index(drop=True, inplace=True)
            weibo.drop(del_index, inplace=True)
            self.weibo = weibo
            self.wbnr = list(self.weibo['jieba_stopword'])
            self.labels = list(self.weibo['province'])

    def labeled_fasttext(self):
        model_file = os.path.join(self.home_data, 'fasttext.model')
        if os.path.exists(model_file):
            pass
        else:
            train_test = len(self.wbnr) * 0.7
            logging.info("训练集70，测试集30,一共%s" % str(len(self.wbnr)))
            f_train = open(self.home_data+"fasttext_train.txt", 'w+', encoding='utf-8')
            f_test = open(self.home_data+"fasttext_test.txt", 'w+', encoding='utf-8')
            for i, (sentence, label) in enumerate(zip(self.wbnr, self.labels)):
                line = sentence + '\t__label__' + label + '\n'
                if i < train_test:
                    f_train.write(line)
                else:
                    f_test.write(line)
            f_train.close()
            f_test.close()

    def fasttext_train(self):
        model_file = os.path.join(self.home_data, 'fasttext.model')
        if os.path.exists(model_file):
            print("Fasttext模型已经存在，直接载入")
            classifier = ft.load_model(self.home_data + 'fasttext.model')
        else:
            print("Fasttext模型不存在，训练")
            import fastText.FastText as ft
            classifier = ft.train_supervised(self.home_data+"fasttext_train.txt")  # 训练模型

            model = classifier.save_model(self.home_data+'fasttext.model')  # 保存模型
            classifier = ft.load_model(self.home_data+'fasttext.model')  # 导入模型
        result = classifier.test(self.home_data+"fasttext_test.txt")  # 输出测试结果
        labels = classifier.get_labels()  # 输出标签
        print("测试实例数", result[0])  # 实例数
        print("准确率", result[1])  # 全部的准确率
        print("召回率", result[2])  # 召回率
        logging.info('测试实例数 %s' % str(result[0]))
        logging.info('准确率 %s' % str(result[1]))
        logging.info('召回率 %s' % str(result[2]))




if __name__ == '__main__':
    home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"
    colname = ['jieba_stopword', 'province', '用户ID', '用户名', 'aite', 'content_all']
    import fastText.FastText as ft
    ft = Fasttext_test(home_data=home_data, colname=colname,
                       input_df="16w_user_340wweibo_merge_province_label164016_delaite_25276.csv")
    ft.load_data()
    ft.labeled_fasttext()
    ft.fasttext_train()