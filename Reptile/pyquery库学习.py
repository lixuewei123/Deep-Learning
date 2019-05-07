# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:01:11 2018

@author: LXW
"""

#html = """"
#    <div>
#        <ul>
#             <li class="item-0">first item</li>
#             <li class="item-1"><a href="link2.html">second item</a></li>
#             <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#             <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#             <li class="item-0"><a href="link5.html">fifth item</a></li>
#         </ul>
#     </div>
#    """
#from pyquery import PyQuery as pq
#doc = pq(html)#doc实际上就称为了一个pyquery对象
#print(doc('li'))#选ID加#    class   .     标签什么也不加
#
#

#URL初始化
#from pyquery import PyQuery as pq
#doc = pq(url='http://www.baidu.com')
#print(doc('head'))
##通过文件初始化
#doc = pq(filename = 'demo.html')


#CSS解析器
#html = """
#    <div id="container">
#        <ul class="list">
#             <li class="item-0">first item</li>
#             <li class="item-1"><a href="link2.html">second item</a></li>
#             <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#             <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#             <li class="item-0"><a href="link5.html">fifth item</a></li>
#         </ul>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc = pq(html)
#print(doc('#container .list li'))#用空格来代表嵌套关系


#查找元素
#html = """
#    <div id="container">
#        <ul class="list">
#             <li class="item-0">first item</li>
#             <li class="item-1"><a href="link2.html">second item</a></li>
#             <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#             <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#             <li class="item-0"><a href="link5.html">fifth item</a></li>
#         </ul>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc = pq(html)
#items = doc('.list')
#print(type(items))
#print(items)
#lis = items.find('li')
#print(type(lis))
#print(lis)
#lis = items.children()#选择items的子元素
#print(type(lis))
#print(lis)
#对子元素传入参数查找
#lis= items.children('.active')# 找出items的子元素，然后找它子元素class为active的
#print(lis)
#父元素，每个标签只有一个父元素
#html = """
#    <div id="container">
#        <ul class="list">
#             <li class="item-0">first item</li>
#             <li class="item-1"><a href="link2.html">second item</a></li>
#             <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#             <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#             <li class="item-0"><a href="link5.html">fifth item</a></li>
#         </ul>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc = pq(html)
#items = doc('.list')
#container = items.parent()#拿到他的父元素
##print(type(container))
##print(container)
##类似的parents方法，查找所有的祖先节点
##兄弟元素
#li=doc('.list .item-0.active')
#print(li.siblings())
#print(li.siblings('.active'))







#***************遍历**********************************************
#html= """"
#    <div class="wrap">
#        <div id="container">
#            <ul class="list">
#                 <li class="item-0">first item</li>
#                 <li class="item-1"><a href="link2.html">second item</a></li>
#                 <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#                 <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#                 <li class="item-0"><a href="link5.html">fifth item</a></li>
#             </ul>
#         </div>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc= pq(html)
#li = doc('.item-0.active')
#print(li)
#lis = doc('li').items()##########生成器，可用for循环提取
#print(type(lis))
#for li in lis:
#    print(li)


#获取信息
#获取属性呢
#a= doc('.item-0.active a')
#print(a)
#print(a.attr('href'))
#print(a.attr.href)#获取属性
#print(a.text())#获取文本
##获取HTML
#print(a.html())














#***********************DOM操作*******************
#addclass      removeclass
#html= """"
#    <div class="wrap">
#        <div id="container">
#            <ul class="list">
#                 <li class="item-0">first item</li>
#                 <li class="item-1"><a href="link2.html">second item</a></li>
#                 <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#                 <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#                 <li class="item-0"><a href="link5.html">fifth item</a></li>
#             </ul>
#         </div>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc= pq(html)
#li = doc('.item-0.active')
#print(li)
#li.removeClass('active')
#print(li)
#li.addClass('active')
#print(li)
#属性    CSS
#html= """"
#    <div class="wrap">
#        <div id="container">
#            <ul class="list">
#                 <li class="item-0">first item</li>
#                 <li class="item-1"><a href="link2.html">second item</a></li>
#                 <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
#                 <li class="item-1 active"><a href="link4.html">fourth item</a></li>
#                 <li class="item-0"><a href="link5.html">fifth item</a></li>
#             </ul>
#         </div>
#     </div>
#"""
#from pyquery import PyQuery as pq
#doc= pq(html)
#li = doc('.item-0.active')
#print(li)
#li.attr('name','link')#给这个标签添加一个name=link的属性
#print(li)
#li.css('fon-size','14px')#给这个标签添加一个    CSS    fon-size = 14px
#print(li)
#remove    方法
#  remove  移除
####   标签的子标签    *。find('p').remove()
#伪类选择器
html= """"
    <div class="wrap">
        <div id="container">
            <ul class="list">
                 <li class="item-0">first item</li>
                 <li class="item-1"><a href="link2.html">second item</a></li>
                 <li class="item-0 active"><a href="link3.html"><span class="bold">third item</span></a></li>
                 <li class="item-1 active"><a href="link4.html">fourth item</a></li>
                 <li class="item-0"><a href="link5.html">fifth item</a></li>
             </ul>
         </div>
     </div>
"""
from pyquery import PyQuery as pq
doc= pq(html)
li = doc('li:first-child')#获取第一个li标签
li = doc('li:last-child')#获取最后一个li标签
li = doc('li:nth-child(2)')#获取制定的li标签
li = doc('li:gt(2)')#获取序号比2大的li标签
li = doc('li:nth-child(2n)')#获取偶数为的lI标签
li = doc('li:contains(seconde)')#根据文本来匹配的