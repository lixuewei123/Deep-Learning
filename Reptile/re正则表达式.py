# -*- coding: utf-8 -*-
#coding=UTF-8
"""
Created on Fri Apr 13 16:06:00 2018


@author: LXW
"""

#re.match函数，依次对我们的正则表达式和目标文件进行匹配
#re.match(pattern,string,flags=0)1正则表达式2目标字符串3匹配模式
#最常规的匹配
import re
content = 'Hello 123 1234 woedf is a xy demo'
result = re.match('^Hello\s\d\d\d\s\d{4}\s.*demo$',content)#^匹配字符串\s空格字符\d数字{重复匹配前面几次}.任意字符*0个或多个表达式$结尾
print(result)
print(result.group())#返回匹配结果
#print(result.span())#输出匹配结果的范围
##泛匹配
#result = re.match('^Hello.*demo$',content)
#print(result)
#print(result.group())#返回匹配结果
#print(result.span())#输出匹配结果的范围
##匹配目标
#import re
#content = 'Hello 1231234 woedf is a xy demo'
#result = re.match('^Hello\s(\d+)\swoedf',content)
#print(result.group(1))#输出第一个小括号里的内容
#print(result.span())
###贪婪匹配
#import re
#content = 'Hello 1231234 woedf is a xy demo'
#result = re.match('^He.*(\d+).*',content)#。*匹配尽可能多的字符
#print(result.group(1))#输出第一个小括号里的内容
#print(result.span())
##非贪婪匹配
#import re
#content = 'Hello 1231234 woedf is a xy demo'
#result = re.match('^He.*?(\d+).*',content)#?尽可能少的字符
#print(result.group(1))#输出第一个小括号里的内容
#print(result.span())
#匹配模型
#.不能匹配换行符，  re.S 可以匹配任意字符
#import re
#content = "Hello 1231234 woedf is a xy demo"
#result = re.match('^He.*?(\d+).*',content,re.S)#?尽可能少的字符
#print(result.group(1))#输出第一个小括号里的内容
#print(result.span())
##转义，特殊字符不能直接匹配，加个/
#import re
#content = 'price is $4.00'#就得加转义字符匹配
#result = re.match('price is \$5\.00',content)
#print(result)
#总结，长字符用.*来代替，最好用非贪婪匹配，尽量使用非常量模式，换行符得加re.S
#re.match 是从第一个字符匹配，如果第一个字符不匹配，就不能完成正常的匹配



#
#***********re.search
#re.search直接搜索整个字符串，并返回第一个成功的匹配
#import re
#content = 'ajsd s f 123445 world demo allfj alf'
#reslut = re.match('f.*?(\d+).*?demo',content)
#print(reslut)#因为第一个字符是不同的，所以直接就不匹配了
#reslut2 = re.search('f.*?(\d+).*?demo',content)
#print(reslut2.group())#成功的匹配到了
#print(reslut2.group(1))
##总结，能用search方法就不用match方法






#**************匹配练习
html = '''<div id=\"songs-list\">\n",
    "    <h2 class=\"title\">经典老歌</h2>\n",
    "    <p class=\"introduction\">\n",
    "        经典老歌列表\n",
    "    </p>\n",
    "    <ul id=\"list\" class=\"list-group\">\n",
    "        <li data-view=\"2\">一路上有你</li>\n",
    "        <li data-view=\"7\">\n",
    "            <a href=\"/2.mp3\" singer=\"任贤齐\">沧海一声笑</a>\n",
    "        </li>\n",
    "        <li data-view=\"4\" class=\"active\">\n",
    "            <a href=\"/3.mp3\" singer=\"齐秦\">往事随风</a>\n",
    "        </li>\n",
    "        <li data-view=\"6\"><a href=\"/4.mp3\" singer=\"beyond\">光辉岁月</a></li>\n",
    "        <li data-view=\"5\"><a href=\"/5.mp3\" singer=\"陈慧琳\">记事本</a></li>\n",
    "        <li data-view=\"5\">\n",
    "            <a href=\"/6.mp3\" singer=\"邓丽君\">但愿人长久</a>\n",
    "        </li>\n",
    "    </ul>\n",
    "</div>'''
import re
result = re.search('<li.*?singer="(.*?)">(.*?)</a>',html,re.S)
if result:
    print(result.group(1),result.group(2),result.span())#*怎么显示中文字符？？？？？？？？？？？？？？？？？？？？
else:
    print("error")
#re.findall 匹配所有的
result2 = re.findall('<li.*?href=\"(.*?)"singer=\"(.*?)">(.*?)</li>',html,re.S)
if result2:
    print(result2.group(1))
else:
    print("error2")
result3 = re.findall('<li.*?>\s*?(<a.*?>)?(</a>)?\s*?</li>',html,re.S)#.代表所有的任意  *代表多个  ？代表是否有
if result3:
    print(result3)







#*******************re.sub
#替换字符串中每一个匹配的字串后返回替换的字符串
#1，传入一个正则表达式，2，要替换成的字符串
#import re
#content = 'extra string 12344 world daf'
#content = re.sub('\d+','',content)#把字符串替换成   ‘’
#print(content)
##用它字符串本身的内容做替换
#import re
#content = 'extra string 12344 world daf'
#content = re.sub('(\d+)',r'\1 521',content)#把字符串替换成   ‘’
#print(content)
#import re
#html = '''<div id=\"songs-list\">",
#   <h2 class="title">经典老歌</h2>",
#   <p class=\"introduction\">",
#        经典老歌列表",
#    </p>\n",
#    <ul id=\"list\" class=\"list-group\">\n",
#        <li data-view=\"2\">一路上有你</li>\n",
#       <li data-view=\"7\">\n",
#            <a href=\"/2.mp3\" singer=\"任贤齐\">沧海一声笑</a>\n",
#        </li>\n",
#        <li data-view=\"4\" class=\"active\">\n",
#            <a href=\"/3.mp3\" singer=\"齐秦\">往事随风</a>\n",
#        </li>\n",
#        <li data-view=\"6\"><a href=\"/4.mp3\" singer=\"beyond\">21212</a></li>\n",
#        <li data-view=\"5\"><a href=\"/5.mp3\" singer=\"陈慧琳\">记事本</a></li>\n",
#        <li data-view=\"5\">\n",
#            <a href=\"/6.mp3\" singer=\"邓丽君\">大</a>\n",
#       </li>\n",
#    </ul>\n",
#   </div>'''
#res = re.sub('<a.*?>|</a>','',html)
#print(res)
#res2 = re.findall('<li.*?>(.*?)</li>',res,re.S)
#print(res2)









#************re.compile
#把正则字符串编译成正则表达式对象
#将一个正则表达式串编译成正则对象，以便于复用该匹配模式
#import re
#content = "Hello 1234 weor"
#pattern = re.compile('Hello.*r',re.S)
#res2 = re.match(pattern,content)
#res = re.match('Hello.*r',content,re.S)
#print(res2)
#print(res)







#***************实战练习
#import requests
#import re
#content = requests.get('https://m.douban.com/book/').text
#print(content)
#pattern = re.compile('<li.*?class="item-title">(.*?)</span>',re.S)
#results = re.findall(pattern,content)
#print(results)