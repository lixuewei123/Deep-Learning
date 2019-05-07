# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:10:02 2018

@author: LXW
"""

#**********requests

import requests
response = requests.get('https://www.baidu.com/')
print(type(response))   #type  是一个class
print(response.status_code)# 状态码是200
print(type(response.text))#类型是   unicode
print(response.text)#  值  类型是字符串
print(response.cookies)#


#****各种请求方式******
#import requests
#requests.get('http://httpbin.org/post')
#requests.put('http://httpbin.org/put')
#requests.delete('http://httpbin.org/delete')
#requests.head('http://httpbin.org/get')
#requests.options('http://httpbin.org/get')



#***********基本get请求
#import requests
#requests.get('http://httpbin.org/get')
##带参数的get请求
#data = {
#        'name':'germey',
#        'age' :22
#        }
#requests.get('http://httpbin.org/get',params= data)#params  参数  用字典的形式来传递get参数



#**************解析json
#import requests
#import json
#response = requests.get("http://httpbin.org/get")
#print (type(response.text))
#print(response.json())
#print(json.loads(response.text))#都是打印json，结果与上一条相同   在一些ajax请求的时候常用
#print(type(response.json()))



#************获取二进制数据
#import requests
#response = requests.get("http://github.com/favicon.ico")
#print (type(response.text),type(response.content))
#print (response.text)
#print (response.content)#response.content 获取图片的二进制内容
##保存图片
#with open('favicon.ico','wb') as f:
#    f.write(response.content)   #把图片写入
#    f.close()



#********添加headers
#import requests
##response = requests.get("http://zhihu.com/explore")  #不加header  错误500
##print (response.text)
#headers = {
#        'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
#        }
#response = requests.get("http://zhihu.com/explore",headers = headers)#加入了我浏览器信息，爬取了知乎的网页信息
#print(response.text)


****************post请求
import requests
data = {'name':'germey','age':'22'}#formdata 表单
#response = requests.post("http://httpbin.org/post",data=data)
#print(response.text)
#增加一个headers，跟get方法一样
headers = {
       'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
        }
response = requests.post("http://httpbin.org/post",data=data,headers = headers)
print(response.json())



#********************response  的属性
#import requests
#response = requests.get("http://www.jianshu.com")
##print(response.text)
#print(type(response.status_code),response.status_code)
#print(type(response.headers),response.headers)
#print(type(response.cookies),response.cookies)
#print(type(response.url),response.url)
#print(type(response.history),response.history)
#状态码判断
#response = requests.get("http://www.jianshu.com")
#exit() if not response.status_code==403 else print("111")



#**************requests  的高级操作
# 文件上传
#import requests
#files = {'file':open('favicon.ico','rb')}
#response = requests.post("http://httpbin.org/post",files = files)#将这个图片上传到这个网址
#print(response.text)
# 获取cookie
#import requests
#response = requests.get("http://www.baidu.com")
#print(response.cookies)
#for key,value in response.cookies.items():
#    print(key + '=' + value)#打印每条cookie的key和value
#cookie是用来做会话维持的，有了cookie就可以维持一个登陆状态
#**********模拟登陆
#import requests
#requests.get('http://httpbin.org/cookies/set/number/123456789')#set 方法设置这么一个网站的cookie
#response = requests.get('http://httpbin.org/coolies')
#print(response.text)
#s = requests.session()#session对象就是将后面的请求打包在一个浏览器里面
#***********证书验证
#import requests
#response = requests.get('https://www.12306.cn',verify=False)#sssl认证的错误, verify=false“不需要验证
###response = requests.get('https://www.12306.cn',cert=('/path/server.crt','/path/key'))#添加证书
#print(response.status_code)
##********代理设置
#import requests
#proxies = {
#        "http":"http://127.0.0.1:9743",
#        "https":"https://127.0.0.1:9743",
#        }# 设置代理的地址
#response = requests.get("https://www.taobao.com",proxies = proxies)
#**************超时的设置
#import requests
#from requests.exceptions import ReadTimeout#引入readtimeout这个类
#try:
#    response = requests.get("https://www.taobao.com",timeout=0.01)
#    print(response.status_code)
#except ReadTimeout:
#    print('Timeout')

#*************认证的设置，，用户名。密码
#import requests
#from requests.auth import HTTPBasicAuth
#response = requests.get('https://passport.weibo.cn/signin/login?entry=mweibo&r=http%3A%2F%2Fweibo.cn%2F&backTitle=%CE%A2%B2%A9&vt=',auth = HTTPBasicAuth('user','123'))
#print(response.status_code)
##print(response.text)

#************异常处理
import requests
from requests.exceptions import ReadTimeout,HTTPError,RequestException
try:
    response =requests.get("http://httpbin.org/get",timeout=0.5)
    print(response.status_code)
except ReadTimeout:
    print('Timeout')
except HTTPError:
    print('Http Error')
except RequestException:
    print('Error')