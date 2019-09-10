# -*- coding:utf-8 -*-
__author__ = 'GIS'

import pandas as pd
import pickle
import sys
import re
import os
import numpy as np
from fanti_jianti.langconv import Converter
import jieba
import pyhanlp
from pyhanlp import *
import logging
import collections


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO,
                    filename="E:/deep_network/GCN Highway My Data/logging.log",
                    filemode='w+')


# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


def extract_ner(ha_ner1):  # 提取 地名和 机构名
    result = []
    for line in ha_ner1:
        line = line.split(',')
        test = []
        for ele in line:
            ele = str(ele)
            if ele[-3:] == '/ns' or ele[-3:] == '/nt':
                test.append(ele)
        result.append(test)
    return result


class Data_Process():
    def __init__(self, home_data, colname, input_df, output_df):
        self.home_data = home_data
        self.colname = colname
        self.input_df = input_df
        self.output_df = output_df

    def read_data(self):
        logging.info('loading the dataset from %s' % self.home_data)
        weibo = pd.read_csv(self.home_data+self.input_df, encoding='utf-8', header=None, error_bad_lines=False)
        weibo.columns = self.colname
        self.weibo = weibo

    def read_stopword(self):
        logging.info('loading the stopwords from %s stop_word.txt' % self.home_data)
        with open(self.home_data+"/stop_words.txt", 'r', encoding='utf-8') as f:
            stopword = f.readlines()
        stopword1 = []
        for word in stopword:
            word = word.strip('\n')
            stopword1.append(word)
        stopword = stopword1
        self.stopword = stopword

    def fanti_jianti(self):         # 添加jianti列
        if 'jianti' not in self.colname:
            weibo = self.weibo
            weibo['微博内容'] = weibo['微博内容'].fillna('')
            weibo['被转发微博内容'] = weibo['被转发微博内容'].fillna('')
            sentences = list(weibo['微博内容'] + weibo['被转发微博内容'])  # 合并微博内容和转发内容
            fanti_jianti_cuowu = 0
            sentences_list = []
            for i, sentence in enumerate(sentences):
                if i%10000 == 0:
                    print(i, fanti_jianti_cuowu, sentence)
                try:
                    sentence = cht_to_chs(sentence)
                    sentences_list.append(sentence)
                except:
                    fanti_jianti_cuowu = fanti_jianti_cuowu+1
                    sentence = sentence
                    sentences_list.append(sentence)
            # sentences = [cht_to_chs(sentence) for sentence in sentences]
            self.weibo['jianti'] = sentences
            self.colname = list(self.weibo.columns)
            logging.info('将微博内容和被转发微博内容合并的繁体字转为简体，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘jianti’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def extract_aite(self):         # 提取@XX,添加一列aite
        if 'aite' not in self.colname:
            self.weibo['jianti'] = self.weibo['jianti'].fillna('')
            pattern = "[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?"
            wbnr = list(self.weibo['jianti'])
            self.weibo['被转发用户名'] = self.weibo['被转发用户名'].fillna('')
            zhuanfa = list(self.weibo['被转发用户名'])
            # wb = []
            # for sentence in wbnr:
            #     sentence = re.sub('//@.*?:', ' ', sentence)  # 将转发的@删除
            #     wb.append(sentence)
            aite_list = []
            for content, reuser in zip(wbnr, zhuanfa):
                content = re.findall(pattern, str(content))
                if reuser != '':
                    content.append(reuser)
                content = [obj.replace('@', '').replace('＠', '').strip(':') for obj in content]
                content = ','.join(content)
                aite_list.append(content)
            self.weibo['aite'] = aite_list
            self.colname = list(self.weibo.columns)
            logging.info('从微博内容和被转发内容中的提取@对象加被转发用户合并为"aite"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘aite’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def extract_label(self):            # 添加label列
        if 'label' not in self.colname:
            self.weibo['jianti'] = self.weibo['jianti'].fillna('')
            pattern = "#(.*?)#"
            wbnr = list(self.weibo['jianti'])
            ob = [re.sub("[\[\]\@\'\ ]", '', str(re.findall(pattern, str(w)))).strip() for w in wbnr]
            self.weibo['label'] = ob
            self.colname = list(self.weibo.columns)
            logging.info('从微博内容中的提取#label#为"label"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘label’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def extract_title(self):            # 添加title列
        if 'title' not in self.colname:
            self.weibo['jianti'] = self.weibo['jianti'].fillna('')
            pattern = "【(.*?)】"
            wbnr = list(self.weibo['jianti'])
            ob = [re.sub("[\[\]\@\'\ ]", '', str(re.findall(pattern, str(w)))).strip() for w in wbnr]
            self.weibo['title'] = ob
            self.colname = list(self.weibo.columns)
            logging.info('从微博内容中的提取【title】为"title"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘title’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def clean_weibo(self):
        if 'wbnr_clean' not in self.colname:
            self.weibo['jianti'] = self.weibo['jianti'].fillna('')
            wbnr = list(self.weibo['jianti'])
            wbnr_clean = []
            for i, wb in enumerate(wbnr):
                # if i % 10000 == 0:
                #     print(i)
                wb = str(wb)
                wb = wb.replace('转发微博', ' ')
                wb = wb.replace('分享图片', ' ')
                wb = wb.replace('分享视频', ' ')
                wb = wb.replace('随拍', ' ')
                wb = re.sub('#.*?#', ' ', wb)
                wb = re.sub('\\[(.+?)\\]', ' ', wb)
                wb = re.sub('[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?', ' ', wb)  # 删除@对象
                #        wb = re.sub('【.*?】', ' ', wb)
                wb = re.sub("[A-Za-z0-9]+", ' ', wb)    # 删除字母和数字
                # wb = re.sub("[^\u4e00-\u9fa5]\,\，", ' ', wb)
                wb = ''.join(re.findall("[\u4E00-\u9FA5\,\，\ \。\.]", wb))
                wbnr_clean.append(wb.strip())
            self.weibo['wbnr_clean'] = wbnr_clean
            self.colname = list(self.weibo.columns)
            logging.info("清洗微博内容，删除'转发微博'、'分享图片'、'分享视频'、'随拍','表情',‘标签’，'@对象'，添加'wbnr_clean'字段，转换后weibo.shape: %s" % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘wbnr_clean’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def jieba_segment(self):
        if 'segment' not in self.colname:
            self.weibo['wbnr_clean'] = self.weibo['wbnr_clean'].fillna('')
            wbnr_clean = list(self.weibo['wbnr_clean'])
            segment_list = []
            for i, wb in enumerate(wbnr_clean):
                # if i % 1000 == 0:
                #     print(i)
                wb = wb.replace('，', ' ').replace('。', ' ').replace(',', ' ').replace('.', ' ')
                seg = jieba.cut(wb, cut_all=False)  # 创建分词对象
                text = ' '.join(seg)
                text = str(text)
                segment_list.append(text.strip())
            self.weibo['segment'] = segment_list
            logging.info('对微博内容jieba分词，添加"segment"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘segment’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def jieba_stopword(self):
        if 'jieba_stopword' not in self.colname:
            self.weibo['segment'] = self.weibo['segment'].fillna('')
            jieba_fenci = list(self.weibo['segment'])
            stopword = self.stopword
            fenci_stopword = []
            for i, sentence in enumerate(jieba_fenci):
                if i % 10000 == 0:
                    print(i)
                sentence = sentence.replace('， ', '').replace('，', '').replace(',', '')
                sentence = sentence.split(' ')
                sentence = [word for word in sentence if word not in stopword]
                sentence = ' '.join(sentence)
                fenci_stopword.append(sentence)
            self.weibo['jieba_stopword'] = fenci_stopword
            self.colname = self.weibo.columns
            logging.info('对jieba分词去停用词，添加"jieba_stopword"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘jieba_stopword’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def hanlp_ner(self):
        if 'ner' not in self.colname:
            HanLP = JClass('com.hankcs.hanlp.HanLP')
            segment = HanLP.newSegment().enablePlaceRecognize(True)
            weibo_clean = list(self.weibo['wbnr_clean'])
            print(len(weibo_clean))
            ha_ner1 = []
            for i, sentence in enumerate(weibo_clean):
                sentence = str(HanLP.segment(str(sentence)))
                ha_ner1.append(sentence)
                # if i % 10 == 0:
                #     print(i, sentence)
            ha_ner = extract_ner(ha_ner1)
            self.weibo['ner'] = ha_ner
            self.colname = self.weibo.columns
            logging.info('对jwbnr_clean提取NER，添加"ner"字段，转换后weibo.shape: %s' % str(self.weibo.shape))
        else:
            logging.info('微博中已经存在‘ner’字段 weibo.shape: %s' % str(self.weibo.shape))
            self.weibo = self.weibo

    def drop_short_sentence(self):
        weibo = self.weibo
        weibo['jieba_stopword'] = weibo['jieba_stopword'].fillna('')
        wbnr = list(weibo['jieba_stopword'])
        del_list = []
        for i, line in enumerate(wbnr):
            line = line.split(' ')
            if len(line) < 3:
                del_list.append(i)
        weibo.reset_index(drop=True, inplace=True)
        weibo.drop(del_list, axis=0, inplace=True)
        weibo.reset_index(drop=True, inplace=True)
        self.weibo = weibo

    def save_data(self):
        weibo = self.weibo
        weibo.to_csv(self.home_data+self.output_df, sep=',', index=False,
                     encoding='utf-8')



if __name__ == '__main__':
    colname = ['是否转发', '用户ID', '用户名', '用户类型', '微博ID', '微博内容', '转发', '评论', '点赞',
       '微博URL', '发文设备', '签到地点', '发文时间', '被转发用户ID', '被转发用户名', '被转发用户类型',
       '被转发微博ID', '被转发微博内容', '被转发微博转发数', '被转发微博评论数', '被转发微博点赞数', '被转发微博URL',
       '被转发微博的设备', '被转发微博的签到', '被转发微博的时间', 'jianti', 'aite', 'label', 'title',
       'wbnr_clean', 'segment', 'jieba_stopword', 'ner']
    home_data = "E:\\deep_network\\GCN Highway My Data\\data\\"

    data = Data_Process(home_data=home_data, colname=colname,
                        input_df="weiboallusers_locate_210000nerusers_data_process.csv",
                        output_df="weiboallusers_locate_210000nerusers_data_process.csv")
    print("read data")
    data.read_data()
    # data.read_stopword()
    # print("繁体转简体 ")
    # data.fanti_jianti()
    # print("save data")
    # data.save_data()
    # print("提取@对象")
    # data.extract_aite()
    # print("提取#标签#")
    # data.extract_label()
    # print("提取【标题】")
    # data.extract_title()
    # print("save data")
    # data.save_data()
    # print("清洗数据")
    # data.clean_weibo()
    # print("结巴分词")
    # data.jieba_segment()
    # print("结巴分词去停用词")
    # data.jieba_stopword()
    # print("save data")
    # data.save_data()
    # print("Hanlp地理命名实体识别")
    # data.hanlp_ner()
    # print("save data")
    # data.save_data()
    # logging.info('数据对象的colname: %s' % str(data.weibo.columns))
    data.drop_short_sentence()
    data.save_data()
    logging.info('数据对象的colname: %s' % str(data.weibo.columns))



