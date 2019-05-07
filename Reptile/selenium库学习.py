# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:46:17 2018

@author: LXW
"""

from selenium import webdriver        #浏览器驱动对象
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
#browser = webdriver.Firefox()         #webdriver.浏览器   声明浏览器对象
#driver = webdriver.Firefox()
browser = webdriver.Chrome()
#browser = webdriver.Firefox()
try:
    browser.get('https://www.baidu.com')
    input = browser.find_element_by_id('kw')  # 调用find_element_by_id         找出ID为kw的元素
    input.send_keys('Python')   #调用send_keys     向元素里发送一些键      
    input.send_keys(Keys.ENTER)
    wait = WebDriverWait(browser,10)
    wait.until(EC.presence_of_element_located((By.ID,'content_left')))
    print(browser.current_url)#百度搜索python 
    print(browser.get_cookies())
    print(browser.page_source)
finally:
    browser.close()












######################详细介绍
#声明浏览器内容
from selenium import webdriver
browser = webdriver.Chrome()
browser = webdriver.firefox()
browser = webdriver.Edge

#访问页面
from selenium import webdriver
browser = webdriver.Firefox()
browser.get('https://www.taobao.com')
print(browser.page_source)
browser.close()


from selenium import webdriver
#browser = webdriver.Firefox()
browser = webdriver.Chrome()
browser.get('https://www.taobao.com')
input_first = browser.find_element_by_id('q')
input_second = browser.find_element_by_css_selector('#q')
input_third = browser.find_element_by_xpath('//*[@id="q"]')
print(input_first,input_second,input_third)
browser.close()

#通用的查找方式
#from selenium import webdriver
#from selenium.webdriver.common.by import By#使用By。查找
#browser = webdriver.Chrome()
#browser.get('https://www.taobao.com')
#input_first = browser.find_element(By.ID,'q')#用By.ID 查  查的ID=q
#print(input_first)
#browser.close()
#



#查找多个元素       find_elements
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.get('https://taobao.com')
#lis = browser.find_elements_by_css_selector('.service-bd li')#选择多个元素就是多加个s的去呗     选出来service-bd里所有的li标签
#print(lis)
#browser.close()







#元素交互操作            先获取特定的元素，进行交互
#from selenium import webdriver
#import time
#browser = webdriver.Chrome()
#browser.get('http://taobao.com')
#input = browser.find_element_by_id('q')#id=q   就是 淘宝页面的输入框
#input.send_keys('iPhone')#在输入框输入iPhone
#time.sleep(1)#等待1秒
#input.clear()
#input.send_keys('ipad')
#button = browser.find_element_by_class_name('btn-search')#就是选中了搜索按钮
#button.click()#点击搜索



#交互动作        网页拖拽
from selenium import webdriver
from selenium.webdriver import ActionChains
browser = webdriver.Chrome()
url = 'http://www.runoob.com/try/try.php?filename=jqueryui-api-droppable'
browser.get(url)
browser.switch_to.frame('iframeResult')
source = browser.find_element_by_css_selector('#draggable')
target = browser.find_element_by_css_selector('#droppable')
actions = ActionChains(browser)
actions.drag_and_drop(source,target)
actions.perform()





#执行js
#通过传入js代码，对网页进行操作，可以弥补没有交互api时
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.get('https://www.zhihu.com/explore')
#browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')#通过execute_script 传入js代码    进度条从顶拉倒低
#browser.execute_script('alert("ahf;ha")')#警告信息








#获取元素信息
#获取属性
#from selenium import webdriver
#browser = webdriver.Chrome()
#url = 'https://www.zhihu.com/explore'
#browser.get(url)
#logo = browser.find_element_by_id('zh-top-link-logo')#知乎 的logo    
#print(logo)
#print(logo.get_attribute('class'))#get_attribute     获取属性
#获取文本值
#from selenium import webdriver
#browser = webdriver.Chrome()
#url = 'https://www.zhihu.com/explore'
#browser.get(url)
#input = browser.find_element_by_class_name('zu-top-add-question')
#print(input.text)#获取文本
#print(input.id)#获取ID
#print(input.location)#获取location 位置
#print(input.tag_name)#获取标签名
#print(input.size)#获取大小





#############Frame
#frame出现比较频繁  相当于一个独立的网页    在父集的frame里面查找子集frame    必须切换到frame里面
#怎么切换父元素frame
#import time
#from selenium import webdriver
#from selenium.common.exceptions import NoSuchElementException
#browser = webdriver.Chrome() 
#url = 'http://www.runoob.com/try/try.php?filename=jqueryui-api-droppable'
#browser.get(url)
#browser.switch_to_frame('iframeResult')
#source = browser.find_element_by_css_selector('#draggable')
#print(source)
#try:
#    logo = browser.find_element_by_class_name('logo')
#except NoSuchElementException:
#    print("no")
#browser.switch_to.parent_frame()###获取它 的父元素  frame
#print(logo)
#print(logo.text)



#wait  
#隐式等待
##当使用隐式等待执行测试的时候，如果webdriver没有在DOM中找到元素，将继续等待，超出设定时间后则抛出找不到元素的异常，换句话说，当查找元素或元素并没有立即出现时，隐式等待将等待一段时间再查找DOM，默认的时间是0
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.implicitly_wait(10)#如果没有加载出来，再额外的等待10秒  如果加载出来了，就不会等待
#browser.get('https://www.zhihu.com/explore')
#input = browser.find_element_by_class_name('zu-top-add-question')
#print(input)

#显示等待     制定一个等待条件    指定一个等待时间
#from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
#browser = webdriver.Chrome()
#browser.get('https://www.taobao.com/')
#wait = WebDriverWait(browser,10)
#input = wait.until(EC.presence_of_all_elements_located(By.ID,'q'))#等待搜索框出现
#button = wait.until(EC.element_to_be_clickable(By.CSS_SELECTOR,'.btn-search'))#等待搜索按钮可以点击
#print(input,button)



#浏览器的前进后退
#import time
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.get('https://www.baidu.com/')
#browser.get('https://www.taobao.com/')
#browser.get('https://www.python.org/')
#browser.back()#后退一步
#time.sleep(1)#等待1秒
#browser.forward()#前进一步
#browser.close()#关闭浏览器



#cookies    登录什么的
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.get('https://www.zhihu.com/explore')
#print(browser.get_cookies())
#browser.add_cookie({'name':'name','domain':'www.zhihu.com','value':'germey'})
#print(browser.get_cookies())
#browser.delete_all_cookies()
#print(browser.get_cookies())



#选项卡管理
#通过执行一个js打开一个网页的选项卡
#通过模拟按键
#import time
#from selenium import webdriver
#browser = webdriver.Chrome()
#browser.get('https://www.baidu.com')
#browser.execute_script('window.open()')#执行js语句   打开新的选项卡
#print(browser.window_handles)#返回所有的选项卡
#browser.switch_to_window(browser.window_handles[1])#切换到第二个选项卡
#browser.get('https://www.taobao.com')
#time.sleep(1)
#browser.switch_to_window(browser.window_handles[0])#切回第一个选项卡
#browser.get('https://python.org')