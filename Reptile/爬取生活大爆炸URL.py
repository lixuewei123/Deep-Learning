# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:48:42 2018

@author: GIS
"""

import urllib.parse
import time
import datetime
import re            
import os    
import sys  
import codecs  
import shutil
import urllib.request, urllib.parse, urllib.error 
from selenium import webdriver        
from selenium.webdriver.common.keys import Keys        
import selenium.webdriver.support.ui as ui        
from selenium.webdriver.common.action_chains import ActionChains
import xlwt
import requests

######################第一季
response = requests.get('http://www.msj1.com/archives/11.html#download')
text = response.text
url = re.findall('<td><a href="(.*?)target="_blank">.*?</a></td>',text,re.S)
print(url)
url1 = url[0:17]
f = open('生活大爆炸第一季.txt','w')
for u in url1:
    f.write(u)
    f.write('\n')
f.close()
####################第二季
response = requests.get('http://www.msj1.com/archives/15.html#download')
text = response.text
url = re.findall('<td><a href="(.*?)target="_blank">.*?</a>.*?</td>',text,re.S)
print(url)
url1 = url[0:23]
print(url1)
f = open('生活大爆炸第2季.txt','w')
for u in url1:
    f.write(u)
    f.write('\n')
f.close()
####################第三季
response = requests.get('http://www.msj1.com/archives/18.html#download')
text = response.text
url = re.findall('<td><a href="(.*?)target="_blank">.*?</a>.*?</td>',text,re.S)
print(url)
url2 = list()
for u in url:
    url2.append(urllib.parse.unquote(u))
url1 = url2[0:23]
print(url1)
####################第四季
response = requests.get('http://www.msj1.com/archives/22.html#download')
text = response.text
url = re.findall('<td><a href="(.*?)target="_blank">.*?</a>.*?</td>',text,re.S)
print(url)
url2 = list()
for u in url:
    url2.append(urllib.parse.unquote(u))
print(url2)
url1 = url2[0:24]
print(url1)


#登录百度云盘
#driver = webdriver.Chrome()
##driver = webdriver.Firefox()
#
#
#def login(username,password):
#        try:
#            #输入用户名/密码登录
#            print('准备登陆yunpan.cn网站...')
#            driver.get('https://pan.baidu.com/')
#            elem_go = driver.find_element_by_id('TANGRAM__PSP_4__footerULoginBtn')
#            elem_go.click()
#            elem_user = driver.find_element_by_name("userName")
#            elem_user.send_keys('18363945218') #用户名
#            elem_pwd = driver.find_element_by_name("password")
#            elem_pwd.send_keys('lixuewei123')  #密码
#            elem_sub = driver.find_element_by_id('TANGRAM__PSP_4__submit')
#            elem_sub.click()
#            elem_know = driver.find_element_by_class_name('tip-button')
#            elem_know.click()
#            for cookie in driver.get_cookies(): 
#                print(cookie)
#                for key in cookie:
#                    print(key, cookie[key]) 
#        except Exception as e:      
#            print("Error: ",e)
#    
#def download():
#    try:
#        elem_new = driver.find_element_by_id('_disk_id_2')
#        elem_new.click()
#        elem_url = driver.find_element_by_id('share-offline-link')
#        elem_url.send_keys(url1[0])#输入URL
##        elem_done = driver.find_element_by_class_name('g-button-right')
##        elem_done = driver.find_element_by_class_name('g-button-blue')
##        elem_done.click()
#        print("点击确定")
#    except Exception as e:
#        print(e)
        
        

#driver = webdriver.Firefox()
driver = webdriver.Chrome()
print('准备登陆yunpan.cn网站...')
driver.get('https://pan.baidu.com/')
elem_go = driver.find_element_by_id('TANGRAM__PSP_4__footerULoginBtn')
elem_go.click()
elem_user = driver.find_element_by_name("userName")
elem_user.send_keys('15621483360') #用户名
elem_pwd = driver.find_element_by_name("password")
elem_pwd.send_keys('xinyu520')  #密码
elem_sub = driver.find_element_by_id('TANGRAM__PSP_4__submit')
elem_sub.click()
elem_know = driver.find_element_by_class_name('tip-button')
elem_know.click()
print("请点击离线下载")


def login():
    driver.get('https://pan.baidu.com/')
    time.sleep(3)
    elem_go = driver.find_element_by_id('TANGRAM__PSP_4__footerULoginBtn')
    elem_go.click()
    elem_user = driver.find_element_by_name("userName")
    elem_user.send_keys('18363945218') #用户名
    elem_pwd = driver.find_element_by_name("password")
    elem_pwd.send_keys('lixuewei123')  #密码
    time.sleep(3)
    elem_sub = driver.find_element_by_id('TANGRAM__PSP_4__submit')
    elem_sub.click()
    time.sleep(3)
    elem_know = driver.find_element_by_class_name('tip-button')
    elem_know.click()
    print("登录成功")

driver = webdriver.Chrome()
login()
#传入URL
def download(url):
    elem_new = driver.find_element_by_id('_disk_id_2')
    elem_new.click()
    elem_url = driver.find_element_by_id('share-offline-link')
    elem_url.send_keys(url)#输入URL
    elem_done = driver.find_element_by_css_selector('#newoffline-dialog > div.dialog-footer.g-clearfix > a.g-button.g-button-blue')
    elem_done.click()
    time.sleep(10)
url1 = url1[11:15]
print(url1)
for u in url1:
    download(u)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    