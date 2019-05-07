# -*- coding: utf-8 -*-
"""
Created on Sun May 06 09:44:39 2018

@author: LXW
"""

#import requests
#from urllib.parse import urlencode
#from requests.exceptions import RequestException
#
#def get_page_index(offset,keyword):
#    data = {
#            'offset': offset,
#            'format':'json',
#            'keyword': keyword,
#            'autoload': 'true',
#            'count': '20',
#            'cur_tab': 1,
#            'from': 'gallery'    
#            }
#    url= 'https://www.toutiao.com/search_content/?' + urlencode(data)   #可以把字典对象转换成URL的请求参数
#    try:
#        response = requests.get(url)
#        if response.status_code == 200:
#            return response.text
#        return None
#    except RequestException:
#        print('请求失败')
#        return None
#
#
#def main():
#    html = get_page_index(0,'街拍')
#    print(html)
#
#if __name__ == '__main__':
#    main()




#根据url获取详情页的html页面
import requests
#from urllib.parse import urlencode
#from requests.exceptions import RequestException
#from bs4 import BeautifulSoup

#data = {
#        'offset': 0,
#        'format':'json',
#        'keyword': '街拍',
#        'autoload': 'true',
#        'count': '20',
#        'cur_tab': 1,
#        'from': 'gallery'  
#        }
url= 'https://www.toutiao.com/search_content/?offset=0&format=json&keyword=%E8%A1%97%E6%8B%8D&autoload=true&count=20&cur_tab=3&from=gallery'   #可以把字典对象转换成URL的请求参数
res = requests.get(url)
print(res.status_code)
#def get_detail_html(url):
#    try:
#        res = requests.get(url)
#        if res.status_code != 200:
#            print('请求详情页失败')
#            return None
#        print('请求详情页成功')
#        html = res.text
#        return html
#    except RequestException:
#        print('异常发生请求失败')
#        return None
#
#
#根据html文件 抓取标题和urls
def parse_html(html):

    soup = BeautifulSoup(html, 'lxml')
    title = soup.select('title')[0].get_text()  #标题在 title标签下 直接获取即可
    print(title)

    #这个含有urls的json语句 位于 var gallery = 这个之后,所以我们做一个匹配
    pattern = re.compile(' var gallery = (.*?);')
    result = re.search(pattern, html)

    if result:
        # 该json包含2个元素
        # coust : 7
        # sub_images : url的list  这是我们需要查找的  这里面还包含了一个字典
        data = json.loads(result.group(1))
        # 查找sub_images是否存在
        if data and 'sub_images' in data.keys():
            sub_images = data.get('sub_images') #获取这个列表

            #获取所有键值为url的 values
            #images = [item['url'] for item in sub_images]
            images = [item.get('url') for item in sub_images]

            #每一张图片的url 传入 获取下载
            [download_image(url) for url in images]

            #将这一组信息传出去
            return {
                'title' : title,
                'urls' : images
            }