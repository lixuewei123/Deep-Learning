import requests
import pandas as pd




province=[]                          #省份
city=[]                              #市
district=[]                          #区
street=[]                            #街区
street_number=[]                     #街牌号
business=[]                          #商圈
cityCode=[]                          #城市代码
key_list = ['',
            'bi4ATD0i3zHvvGC3y93I6LglhOGUaInn',
            'sOtYZ3Z1mMMht5sSu0nGWMd5YFjowonc',
            'Dy9AAhRegb8juKq7nGYq0yCnTfaCS1xE',
            'mMeXvf03yMyg1v0DYhfMeOUZRF9Regf1',
            'm8nZFIIFtH2vbuC9Dkr8bTTSeCt7p0Cn',
            'XKBAbQv4XRxBpyyBwU7ifRntP2XdquKL',
            'FkQGiIGDH73u1KjoSGwYPdVGCiGu7xFn',
            'FkQGiIGDH73u1KjoSGwYPdVGCiGu7xFn',
            '2tOCuGez6ce0qQSohUzA17IYB2q0cY6m',
            'YTSLoDGw5Wxz0O79EeBVNzRcYcLNgZxS',
            '3xcRhBs2zfpjcs7NT4Bvn96EyKaGjQIE']


key = 'bi4ATD0i3zHvvGC3y93I6LglhOGUaInn'
url = 'http://api.map.baidu.com/geocoder?output=json&key='+ str(key) +'&location=' +str(34.725886)+','+str(112.78253)
requests.get(url).json()
# 速度超级慢，少量数据可以使用。

# 读取数据
weibo = pd.read_csv("E:/lxw_data/data filter/data/weibo_locate_1758707.csv", encoding='utf-8', header=None)
weibo.columns= ['Id', '发文时间_x', '抓取时间戳_x', '微博ID', '用户ID_x', '用户昵称', '地址', '维度', '经度',
       '签到位置', '日期', 'ID', '抓取时间', '抓取时间戳_y', '是否转发', '用户ID_y', '用户名', '用户类型',
       '微博内容', '转发', '评论', '点赞', '微博URL', '发文设备', '签到地点', '发文时间_y', '发文时间戳',
       'wbnr_clean', 'segment', 'aite', 'aitel', 'label']
lat = list(weibo['维度'])
lng = list(weibo['经度'])


def get_geocode(lat, lng, key_list):
    
    province=[]                          #省份
    city=[]                              #市
    district=[]                          #区
    street=[]                            #街区
    street_number=[]                     #街牌号
    business=[]                          #商圈
    cityCode=[]                          #城市代码
        
    error_list = []
    for i, (wei, jing) in enumerate(zip(lat, lng)):
        
        if i % 290000 == 0:
            del key_list[0]
            print(len(key_list),key_list)
            
        for j, key in enumerate(key_list):
                
            url = 'http://api.map.baidu.com/geocoder?output=json&key='+ str(key) +'&location=' +str(wei)+','+str(jing)
            try:
                answer = requests.get(url).json()
                province.append(answer['result']['addressComponent']['province'])
                city.append(answer['result']['addressComponent']['city'])
                district.append(answer['result']['addressComponent']['district'])
                street.append(answer['result']['addressComponent']['street'])
                street_number.append(answer['result']['addressComponent']['street_number'])
                business.append(answer['result']['business'])
                cityCode.append(answer['result']['cityCode'])
                
                a1 = answer['result']['addressComponent']['province']
                a2 = answer['result']['addressComponent']['city']
                a3 = answer['result']['addressComponent']['district']
                a4 = answer['result']['addressComponent']['street']
                a5 = answer['result']['addressComponent']['street_number']
                a6 = answer['result']['business']
                a7 = answer['result']['cityCode']
                
                address = pd.DataFrame([[str(wei),str(jing),a1,a2,a3,a4,a5,a6,a7]],
                               columns=['lat','lng','province','city','district','street','street_number','business','cityCode'])
                break
            
            except:
                province.append('')
                city.append('')
                district.append('')
                street.append('')
                street_number.append('')
                business.append('')
                cityCode.append('')
                
                address = pd.DataFrame([[str(wei),str(jing),'','','','','','','']],
                               columns=['lat','lng','province','city','district','street','street_number','business','cityCode'])
                if j == 10:
                    print("无效经纬度", cor)
                    error_list.append(cor)
                continue
            
        if i%10==0:
             print(i)

        address.to_csv("E:/lxw_data/data filter/data/address_baidu.csv", encoding='utf-8',header=False,index=False,mode='a')
        
        
    return province,city,district,street,street_number,business,cityCode




get_geocode(34.725886, 112.78253, key_list)
