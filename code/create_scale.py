
# %%
from selenium import webdriver
import time
from bs4 import BeautifulSoup as bs
from selenium.webdriver.chrome.options import Options
import pymysql

# %%
def get_etfs(db_name,type_name_list):
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()
    etf = []
    for type_number in range(len(type_name_list)):
        sql ="select code from `record` where (type =  '"+ type_name_list[type_number] +"')"
        cursor.execute(sql)
        result_select = cursor.fetchall()
        db.commit()
        # print(result_select[0][0])
        temp = str(result_select[0][0])
        temp_list = temp.split(' ')
        for i in range(len(temp_list)):
            etf.append(temp_list[i])
    return etf

# %%
db_name = 'my_etf'
# list_etf = ['US_etf']
list_etf = ['TWO_etf']
etf = get_etfs(db_name,list_etf)
print(len(etf))

# %%
#輸入Mysql前的資料，並且先都設定為0
# name = []
# create = []
# area = []
# buy_type = []
# style = []
# discount = []
# interest = []
# interest_freq = []
# manage = []
# cost = []
# y_std = []
# trace_index =[]
# base_index = []
# scale = []
db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
cursor = db.cursor()

# raw_data是存一個標的全部的基本資料，之後再慢慢切，找到標的性質的位置後再存入，存入後記得把全部的值變回''
raw_data = []
for i in range(1000):
    raw_data.append('')

# etf = ['SPY','IVV']
# 不要把chrome打開的指令
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# 網頁共同的部分
url1 = 'https://www.moneydj.com/ETF/X/Basic/Basic0004.xdjhtm?etfid='

#開啟瀏覽器並進入網頁
driver = webdriver.Chrome("D:/Alia/Documents/asset allocation/chromedriver_win32/chromedriver.exe", chrome_options = chrome_options)

#根據ETF的list重複執行抓取資料的動作
for i in range(len(etf)):
    print(i,etf[i])
    try : 
        url = url1 + etf[i]
        driver.get(url)
        time.sleep(3)
    except:
        print('need to take a break')
        # name.append('0')
        # create.append('0')
        # area.append('0')
        # buy_type.append('0')
        # discount.append('0')
        # interest.append('0')
        # interest_freq.append('0')
        # cost.append('0')
        # y_std.append('0')
        # style.append('0')
        # trace_index.append('0')
        # base_index.append('0')
        # manage.append('0')
        # scale.append('0')
        # 把全部的值變回''
        for i in range(1000):
            raw_data.append('')
        time.sleep(3)
    try :
        soup = bs(driver.page_source,"html.parser")
        # click
        start_search_btn = driver.find_element_by_id("sshow")
        start_search_btn.click()
        time.sleep(3)
        soup = bs(driver.page_source,"html.parser")
        content = driver.find_elements_by_xpath("//td[@class='al']")
        a = 0
        for u in content:
            # print(type(u.text))
            # print(u.text)
            raw_data[a] = u.text
            a = a+1
        # print(raw_data)
        scale = raw_data[18]
        # name.append(raw_data[10])
        # create.append(raw_data[16])
        # scale.append(raw_data[18])
        # area.append(raw_data[26])
        # buy_type.append(raw_data[24])
        # discount.append(raw_data[25])
        # interest.append(raw_data[29])
        # interest_freq.append(raw_data[27])
        # cost.append(raw_data[30])
        # y_std.append(raw_data[31])
        # style.append(raw_data[22])
        # trace_index.append(raw_data[39])
        # base_index.append( raw_data[40])
        # manage.append(raw_data[28])
        # 把全部的值變回''
        for a in range(1000):
            raw_data.append('')
        time.sleep(3)

        sql0 = "UPDATE detail SET 規模 = '"+ scale +"' WHERE name = '" +etf[i]+"'; "
        print(sql0)
        try:
            cursor.execute(sql0)
            db.commit()
            print("Data are successfully inserted into detail")
        except Exception as e:
            db.rollback()
            print("Exception Occured : ", e)
        # break
    except:
        continue
    
# close the driver
driver.close()


# %%

# print('規模')
# print(scale)
# print(len(scale))

# print("etf全名")
# print(name)
# print(len(name))

# print("創立時間")
# print(create)
# print(len(create))

# print("投資區域")
# print(area)
# print(len(area))

# print("投資標的")
# print(buy_type)
# print(len(buy_type))

# print("投資風格")
# print(style)
# print(len(style))

# print("折溢價")
# print(discount)
# print(len(discount))

# print("殖利率")
# print(interest)
# print(len(interest))

# print("配息頻率")
# print(interest_freq)
# print(len(interest_freq))

# print("管理費")
# print(manage)
# print(len(manage))

# print("總費用率")
# print(cost)
# print(len(cost))

# print("年化標準差")
# print(y_std)
# print(len(y_std))

# print("追蹤指數")
# print(trace_index)
# print(len(trace_index))

# print("基準指數")
# print(base_index)
# print(len(base_index))

# %%
