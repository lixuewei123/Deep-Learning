# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:37:54 2018

@author: LXW
"""

from bs4 import BeautifulSoup
html = """
    <html><head><title>The Dormouse's story</title></head>,
    <body>,
    <p class="title" name="dromouse"><b>The Dormouse's story</b></p>,
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
    <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
    <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class=\"story\">...</p>
    """
#Beautifulsoup  解析
soup = BeautifulSoup(html,'lxml')
print(soup.prettify())#格式化HTML标签
# print(soup.title.string)
# #
#
# #
# ##标签选择器
# ##选择元素
# from bs4 import BeautifulSoup
# html = """
#     <html><head><title>The Dormouse's story</title></head>,
#     <body>,
#     <g class="title" name="dromouse"><b>The Dormouse's story</b></g>
#     <p class="story">Once upon a time there were three little sisters; and their names were
#     <a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
#     <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
#     <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
#     and they lived at the bottom of a well.</p>
#     <p class=\"story\">...</p>
#     """
# soup = BeautifulSoup(html,'lxml')
# print(soup.title)#title标签
# print(type(soup.title))#title标签的类型
# print(soup.head)#head标签
# print(soup.g)#P标签，，，只返回第一个结果
# #获取名称
# print(soup.title.name)
# print(soup.g.attrs['name'])#attrs    获取标签内的内容
# print(soup.g['name'])
# print(soup.g.string)#.string获取标签的内容
# print(soup.head.title.string)#层层嵌套
# #
#
#
# #子节点和子孙节点
# html = """
#     <html>
#         <head>
#             <title>The Dormouse's story</title>
#         </head>
#         <body>
#             <p class="story">
#                 Once upon a time there were three little sisters; and their names were
#                 <a href="http://example.com/elsie" class="sister" id="link1">
#                     <span>Elsie</span>
#                 </a>
#                 <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
#                 and
#                 <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
#                 and they lived at the bottom of a well.
#             </p>
#             <p class="story">...</p>
#     """
# from bs4 import BeautifulSoup
# soup= BeautifulSoup(html,'lxml')
# print(soup.p.contents)#p标签里面的子节点就是a标签。  把所有的子节点用列表的形式返回、
# print(soup.p.children)#是一个迭代器对象，内容需要用一个循环才能取到
# for i,child in enumerate(soup.p.children):#enumerate   这个方法会返回节点的内容和索引，  i接受索引，child接收内容
#     print(i,child)
# print(soup.p.descendants)#获取它的所有的子孙节点
# for i,child in enumerate(soup.p.descendants):
#     print(i,child)
# print(soup.a.parent)#获取父节点
# print(soup.a.parents)#获取祖先节点
# print(list(enumerate(soup.a.parents)))
# print(list(enumerate(soup.a.next_siblings)))#获取后面的兄弟节点
# print(list(enumerate(soup.a.previous_siblings)))#获取前面的兄弟节点
#
#
#
#
#
# #标准选择器
# #可根据标签名，属性，内容查找文档
# find——all(name,attrs,recurive,text,**kwargs)
# from bs4 import BeautifulSoup
# html = """
#     <div class="panel">
#         <div class="panel-heading">
#             <h4>Hello</h4>
#         </div>
#         <div class="panel-body">
#             <ul class="list" id="list-1">
#                 <li class="element">Foo</li>
#                 <li class="element">Bar</li>
#                 <li class="element">Jay</li>
#             </ul>
#             <ul class="list list-small" id="list-2">
#                 <li class="element">Foo</li>
#                 <li class="element">Bar</li>
#             </ul>
#         </div>
#     </div>
#     """
# soup = BeautifulSoup(html,'lxml')
# print(soup.find_all('ul'))#根据标签名查找哦
# print(type(soup.find_all('ul')[0]))
# for ul in soup.find_all('ul'):#把ul标签里的，li标签再取出来
#     print(ul.find_all('li'))
# #attrs按照属性查询
# print(soup.find_all(attrs={'id':'list-1'}))#用字典格式来查找属性      标签
# print(soup.find_all(id='list-1'))
# print(soup.find_all(class_="element"))#class比较特殊，在python里是有定义，所以加了下划线传输
# #text
# #通过文本内容
# print(soup.find_all(text='Foo'))
# #find 方法
# #find返回单个元素，find_all返回所有元素，用法跟find_all完全一样，，，，返回匹配的第一个元素
# print(soup.find('ul'))
#
#







#CSS选择器
#select()方法，直接传入CSS选择器
#class就在前面加一个.       标签前面不加内容       id用#
# html = """
#     <div class="panel">
#         <div class="panel-heading">
#             <h4>Hello</h4>
#         </div>
#         <div class="panel-body">
#             <ul class="list" id="list-1">
#                 <li class="element">Foo</li>
#                 <li class="element">Bar</li>
#                 <li class="element">Jay</li>
#             </ul>
#             <ul class="list list-small" id="list-2">
#                 <li class="element">Foo</li>
#                 <li class="element">Bar</li>
#             </ul>
#         </div>
#     </div>
#     """
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(html,'lxml')
# print(soup.select('.panel .panel-heading'))#class   .
# print(soup.select('ul li'))#标签     直接用名字
# print(soup.select('#list-2 .element'))#id  用#
# print(type(soup.select('ul')[0]))
# for ul in soup.select('ul'):
#     print(ul.select('li'))
# #获取标签的属性，内容
# for ul in soup.select('ul'):
#     print(ul['id'])
#     print(ul.attrs['id'])
# #获取内容
# for ul in soup.select('li'):
#     print(li.get_text())