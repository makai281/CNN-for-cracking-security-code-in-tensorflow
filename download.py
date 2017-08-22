import requests
import time


wb = requests.Session()

wb.headers.update({
    'User-Agent':'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36',
})


cookies = {'JSESSIONID': '505BDE0E950089EAAF4E989FC36C7543;'}
url = 'http://openlaw.cn/login.jsp?returnTo=%2Fuser%2F'
r=wb.get(url, cookies = cookies)

for x in range(1, 100001):
    #url = 'http://openlaw.cn/Kaptcha?v=dc8707e112574d5eb38fa6d4dd5d1016&'  + str(int(time.time()))
    time_stamp = str(int(time.time()))
    url = 'http://openlaw.cn/Kaptcha?v='  + time_stamp
    r=wb.get(url, cookies = cookies, headers = {
        'Referer':'http://openlaw.cn/login.jsp?returnTo=%2Fuser%2F',
        'Accept':'image/*,*/*;q=0.8',
    })
    ff=open('./download/'+str(x).rjust(5,'0')+'.png', 'wb')
    ff.write(r.content)
    ff.close()


