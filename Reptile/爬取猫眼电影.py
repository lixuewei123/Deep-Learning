# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:08:09 2018

@author: LXW
"""

import json
from multiprocessing import Pool#多进程 
import requests
from requests.exceptions import RequestException
import re

def get_one_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None

def parse_one_page(html):



































































    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?data-src="(.*?)".*?name"><a'
                         +'.*?>(.*?)</a>.*?star">(.*?)</p>.*?releasetime">(.*?)</p>'
                         +'.*?integer">(.*?)</i>.*?fraction">(.*?)</i>.*?</dd>', re.S)
    items = re.findall(pattern, html)
    for item in items:
        yield {# 遍历  转换成一个字典
            'index': item[0],
            'image': item[1],
            'title': item[2],
            'actor': item[3].strip()[3:],#  切片  把前三个字符切掉
            'time': item[4].strip()[5:],
            'score': item[5]+item[6]
        }

def write_to_file(content):
    with open('result.txt', 'a', encoding='utf-8') as f:# a 参数代表  直接往后追加      改成中文显示 
        f.write(json.dumps(content, ensure_ascii=False) + '\n')#  ASCII码显示为false
        f.close()

def main(offset):
    url = 'http://maoyan.com/board/4?offset=' + str(offset)  #通过改变offset  动态改变  URL
    html = get_one_page(url)
    for item in parse_one_page(html):
        print(item)
        write_to_file(item)


if __name__ == '__main__':
    pool = Pool()#声明一个进程池
    pool.map(main, [i*10 for i in range(10)])#将数组中的每个参数放到进程池中去运行
    pool.close()
    pool.join()







#**********************************
#import requests
#from requests.exceptions import RequestException
#
#def get_one_page(url):
#    try:
#        response = requests.get(url)
#        if response.status_code == 200:
#            return response.text
#        return None
#    except RequestException:
#        return None
#
#def  main():
#    url = 'http://maoyan.com/board'
#    html = get_one_page(url)
#    print(html)
#
#if __name__ == '__main__':
#    main()