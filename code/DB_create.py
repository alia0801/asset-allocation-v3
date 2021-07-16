# %%
from selenium import webdriver
import time
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import pymysql
import datetime
from selenium.webdriver.chrome.options import Options
import requests
import datetime
from dateutil.parser import parse 
import yfinance as yf
from dateutil.relativedelta import relativedelta
from webdriver_manager.chrome import ChromeDriverManager
# %%

############################################  ETF名稱爬蟲   ############################################
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome("C:/Users/User/Downloads/安裝檔案/chromedriver.exe", chrome_options = chrome_options)
# driver = webdriver.Chrome(ChromeDriverManager().install())
url1 = 'https://etfdb.com/screener/#page='
url2 = '&sort_by=assets&sort_direction=desc'

etf = []
#根據ETF的list重複執行抓取資料的動作
for i in range(20):
    a = i + 1 
    print(a)
    driver.get(url1 + str(a) + url2)
    driver.get(url1 + str(a) + url2)
    time.sleep(3)
    soup = bs(driver.page_source,"html.parser")
    time.sleep(3)
    content = driver.find_elements_by_xpath("//td[@data-th='Symbol']")
    for u in content:
        print(type(u.text))
        print(u.text)
        etf.append(u.text)
    time.sleep(3)
    # close the driver
driver.close()

print(len(etf))
print(etf)


############################################  資料庫開始建立   ############################################

'''
資料表都有了 會更動的只有close
下面是創建資料庫 可以驅動網頁的執行順序
0.找到選股目標 擺入目標array
1.把detail的name先存入
2.由yahoo API 抓取close股價 存入etf_close 以及 寫入 成立年限 成立日期 以及是 股票 還是 ETF的三個欄位資料
3.dj資料爬蟲(總費用率)
4.配息率程式執行
5.其他yahoo資料爬蟲
6.各長年化值
7.資料庫建置完成
'''

# temp = 'SPY,IVV'

# # 輸入標的群 存入etf的list中
# # temp = 'SPY,IVV,VTI,VOO,QQQ,AGG,GLD,VEA,IEFA,BND,VWO,VUG,IWF,LQD,IEMG,VTV,EFA,VIG,IJH,IJR,IWM,VCIT,IWD,VGT,XLK,VO,USMV,IAU,VCSH,BNDX,IVW,HYG,VNQ,VB,ITOT,VYM,BSV,VXUS,VEU,EEM,XLV,TIP,IWB,DIA,SCHX,MBB,IXUS,SHY,SHV,IWR,IGSB,IEF,SCHF,QUAL,VV,GDX,XLF,MUB,TLT,PFF,EMB,IVE,SCHB,XLY,SDY,SLV,GOVT,MDY,BIV,XLP,VT,BIL,JPST,MINT,VBR,RSP,JNK,DVY,IWP,SCHD,VGK,ACWI,SCHP,SCHG,XLI,XLU,DGRO,VMBS,VHT,MTUM,IGIB,IEI,VBK,EFAV,XLC,IWS,GSLC,EWJ,FDN,SCHA'
# temp_arr = temp.split(',')
# etf = []
# for i in range(len(temp_arr)):
#     etf.append(temp_arr[i])
etf = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'VEA', 'IEFA', 'AGG', 'VWO', 'IEMG', 'GLD', 'BND', 'VUG', 'IWM', 'VTV', 'IWF', 'IJR', 'IJH', 'EFA', 'LQD', 'VIG', 'IWD', 'VCIT', 'VO', 'VGT', 'VB', 'VXUS', 'BNDX', 'XLK', 'VCSH', 'ITOT', 'VYM', 'VEU', 'IVW', 'USMV', 'IAU', 'VNQ', 'XLF', 'BSV', 'EEM', 'TIP', 'MBB', 'IXUS', 'IWB', 'SCHX', 'XLV', 'IWR', 'DIA', 'SCHF', 'HYG', 'ARKK', 'IGSB', 'VV', 'QUAL', 'MUB', 'VBR', 'MDY', 'PFF', 'IVE', 'RSP', 'SHY', 'XLY', 'EMB', 'SCHB', 'VT', 'SDY', 'SCHD', 'TLT', 'XLE', 'SHV', 'XLI', 'GDX', 'JPST', 'IWP', 'VBK', 'DVY', 'DGRO', 'BIV', 'VGK', 'VXF', 'ACWI', 'MTUM', 'SCHP', 'MINT', 'SCHA', 'GOVT', 'IEF', 'SLV', 'EWJ', 'VHT', 'VMBS', 'SCHG', 'IWN', 'ESGU', 'IWO', 'BIL', 'JNK', 'IWS', 'XLP', 'SCZ', 'VOE', 'GSLC', 'XLU', 'FDN', 'IEI', 'XLC', 'IGIB', 'IWV', 'IBB', 'VTEB', 'EFAV', 'VTIP', 'VLUE', 'ARKG', 'VOT', 'FVD', 'IUSG', 'EFG', 'VGSH', 'MGK', 'SPDW', 'SPYG', 'TQQQ', 'SCHE', 'IHI', 'SCHZ', 'EFV', 'SCHM', 'SCHV', 'IJK', 'FTCS', 'SPLV', 'EWY', 'SPLG', 'SPYV', 'VFH', 'IUSV', 'XBI', 'USHY', 'SCHO', 'IJS', 'SPSB', 'HYLB', 'OEF', 'NOBL', 'CWB', 'ESGE', 'PGX', 'MCHI', 'EWZ', 'VGIT', 'IYW', 'ICLN', 'BBJP', 'LMBS', 'XLB', 'AAXJ', 'IUSB', 'IJJ', 'ARKW', 'EWT', 'VSS', 'SPIB', 'IJT', 'GDXJ', 'SKYY', 'HDV', 'FPE', 'USIG', 'ACWV', 'SPEM', 'VCLT', 'SHYG', 'FNDX', 'FNDF', 'IGV', 'SPAB', 'BKLN', 'VDC', 'VONG', 'BLV', 'FLOT', 'SOXX', 'EZU', 'FTEC', 'DGRW', 'ICSH', 'INDA', 'VCR', 'FTSM', 'VNQI', 'IXN', 'VPL', 'VOOG', 'FIXD', 'AMLP', 'ISTB', 'TAN', 'IDEV', 'SCHH', 'NEAR', 'PRF', 'BBCA', 'SHM', 'VPU', 'SPTM', 'MOAT', 'FXI', 'VIS', 'EEMV', 'ANGL', 'SMH', 'ESGD', 'SUB', 'BOND', 'IYR', 'IEUR', 'VDE', 'FNDA', 'GLDM', 'XSOE', 'SCHR', 'IDV', 'ACWX', 'SJNK', 'FNDE', 'QLD', 'GUNR', 'DBEF', 'SPMD', 'VTWO', 'KWEB', 'MGV', 'ONEQ', 'SPSM', 'USO', 'VONV', 'EMLC', 'QTEC', 'TFI', 'IAGG', 'IWY', 'CIBR', 'MGC', 'BBEU', 'TOTL', 'XT', 'STIP', 'USSG', 'HYD', 'BBIN', 'IGM', 'VIGI', 'SLYV', 'ESGV', 'VOX', 'KBE', 'SSO', 'PDBC', 'SCHC', 'IGF', 'EWU', 'GSY', 'PBW', 'SPTS', 'EWC', 'JETS', 'VNLA', 'SUSL', 'AIA', 'QCLN', 'SPMB', 'SPTL', 'PCY', 'SRLN', 'IOO', 'ITA', 'RPG', 'DON', 'IQLT', 'LIT', 'SOXL', 'XOP', 'DSI', 'VWOB', 'FXL', 'EWG', 'DLN', 'SPHD', 'SGOL', 'VAW', 'KRE', 'IYH', 'BOTZ', 'FHLC', 'SPTI', 'IXJ', 'SPYD', 'ASHR', 'FAS', 'IGLB', 'GBIL', 'SPHQ', 'FV', 'ARKQ', 'REET', 'HEFA', 'RYT', 'SUSA', 'ARKF', 'FLRN', 'SPIP', 'BAB', 'GVI', 'VGLT', 'BSCM', 'FNDC', 'NFRA', 'GSIE', 'PZA', 'HYLS', 'SLYG', 'XLRE', 'MDYG', 'SLQD', 'EPP', 'FBT', 'HACK', 'RODM', 'BSCL', 'KOMP', 'RDVY', 'JHMM', 'PDP', 'FPX', 'VONE', 'GXC', 'HEDJ', 'GEM', 'ITB', 'HYS', 'AMJ', 'FEZ', 'ICF', 'FMB', 'ROBO', 'PTLC', 'VYMI', 'PRFZ', 'BBAX', 'DEM', 'JKE', 'SH', 'DGS', 'EMLP', 'KBWB', 'TECL', 'RSX', 'USMC', 'EWL', 'ITM', 'PGF', 'UPRO', 'EWA', 'XMLV', 'UVXY', 'QYLD', 'XLG', 'CMF', 'IYT', 'MDYV', 'VOOV', 'IBUY', 'DES', 'ILF', 'VSGX', 'FBND', 'EMQQ', 'IYF', 'IEV', 'AOR', 'DXJ', 'SPXL', 'GNR', 'USRT', 'BSCN', 'JKH', 'IBDM', 'RWO', 'FLCB', 'CLOU', 'MJ', 'TNA', 'ICVT', 'XSLV', 'SLY', 'VRP', 'FLQL', 'SQQQ', 'CWI', 'EWH', 'AOM', 'FXH', 'EWW', 'CQQQ', 'IYY', 'QDF', 'TILT', 'SCHK', 'IBDN', 'FDL', 'QLTA', 'TLH', 'HYMB', 'FXD', 'TDIV', 'IBDO', 'PPLT', 'DBC', 'BBMC', 'PHO', 'DLS', 'FTSL', 'RWR', 'PSK', 'VIOO', 'FDIS', 'IYG', 'XHB', 'TDTT', 'REM', 'IYJ', 'XAR', 'OMFL', 'FNGU', 'WCLD', 'PXH', 'IYC', 'BSCO', 'RPV', 'EDV', 'JMST', 'QQEW', 'AOA', 'IVOO', 'IBDP', 'AGGY', 'IGOV', 'ONLN', 'BSCK', 'URTH', 'FREL', 'IHF', 'PULS', 'XSD', 'IXC', 'SIL', 'XME', 'IPAC', 'SIZE', 'PFFD', 'BAR', 'IVOL', 'IWC', 'USFR', 'FTC', 'NUGT', 'EUFN', 'ACES', 'FUTY', 'PXF', 'SDOG', 'FINX', 'PKW', 'IPAY', 'SPLB', 'RPAR', 'PAVE', 'FEX', 'FLCO', 'BWX', 'EBND', 'EAGG', 'JHML', 'VXX', 'PNQI', 'SRVR', 'PSC', 'UCO', 'FIVG', 'IBDQ', 'BBRE', 'INTF', 'FNCL', 'IWX', 'JPIN', 'BSJL', 'KBA', 'IBDL', 'MOO']

# detail欄位名稱 => 預設要有的 : name 資料起始 資料年限 配息率 
detail_col_list = ['name','etf全名','創立時間', '資料起始', '資料年限', '配息率', '投資區域','投資標的','投資風格','折溢價','殖利率','配息頻率','管理費','總費用率','年化標準差','追蹤指數','基準指數']

createsqltable = """CREATE TABLE IF NOT EXISTS """ + 'detail'  + " (" + " VARCHAR(20),".join(detail_col_list) + " VARCHAR(20))" +  " DEFAULT CHARSET=utf8" + ";"
print(createsqltable)
    
db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()

try:
    cursor.execute(createsqltable)
    db.commit()
    print("create table successfully")
    for item in etf : 
        sql = "INSERT INTO `detail` (`name`) VALUES"
        values = "('%s')"
        sql += values % (item)
        cursor.execute(sql)
        db.commit()
        print("Data are successfully inserted")
except Exception as e:
    db.rollback()
    print("Exception Occured : ", e)


############################################  DJ 網站爬蟲 基本資料頁面   ############################################

#輸入Mysql前的資料，並且先都設定為0
name = []
create = []
area = []
buy_type = []
style = []
discount = []
interest = []
interest_freq = []
manage = []
cost = []
y_std = []
trace_index =[]
base_index = []

# raw_data是存一個標的全部的基本資料，之後再慢慢切，找到標的性質的位置後再存入，存入後記得把全部的值變回''
raw_data = []
for i in range(1000):
    raw_data.append('')


# 不要把chrome打開的指令
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# 網頁共同的部分
url1 = 'https://www.moneydj.com/ETF/X/Basic/Basic0004.xdjhtm?etfid='

#開啟瀏覽器並進入網頁
driver = webdriver.Chrome("C:/Users/User/Downloads/安裝檔案/chromedriver.exe", chrome_options = chrome_options)

#根據ETF的list重複執行抓取資料的動作
for i in range(len(etf)):
    print(i)
    try : 
        url = url1 + etf[i]
        driver.get(url)
        time.sleep(3)
    except:
        print('need to take a break')
        name.append('0')
        create.append('0')
        area.append('0')
        buy_type.append('0')
        discount.append('0')
        interest.append('0')
        interest_freq.append('0')
        cost.append('0')
        y_std.append('0')
        style.append('0')
        trace_index.append('0')
        base_index.append('0')
        manage.append('0')
        # 把全部的值變回''
        for i in range(1000):
            raw_data.append('')
        time.sleep(3)

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
    name.append(raw_data[10])
    create.append(raw_data[16])
    area.append(raw_data[26])
    buy_type.append(raw_data[24])
    discount.append(raw_data[25])
    interest.append(raw_data[29])
    interest_freq.append(raw_data[27])
    cost.append(raw_data[30])
    y_std.append(raw_data[31])
    style.append(raw_data[22])
    trace_index.append(raw_data[39])
    base_index.append( raw_data[40])
    manage.append(raw_data[28])
    # 把全部的值變回''
    for i in range(1000):
        raw_data.append('')
    time.sleep(3)
    
        

# close the driver
driver.close()

print("etf全名")
print(name)
print(len(name))

print("創立時間")
print(create)
print(len(create))

print("投資區域")
print(area)
print(len(area))

print("投資標的")
print(buy_type)
print(len(buy_type))

print("投資風格")
print(style)
print(len(style))

print("折溢價")
print(discount)
print(len(discount))

print("殖利率")
print(interest)
print(len(interest))

print("配息頻率")
print(interest_freq)
print(len(interest_freq))

print("管理費")
print(manage)
print(len(manage))

print("總費用率")
print(cost)
print(len(cost))

print("年化標準差")
print(y_std)
print(len(y_std))

print("追蹤指數")
print(trace_index)
print(len(trace_index))

print("基準指數")
print(base_index)
print(len(base_index))


# cost是總管理費用(%)	0.03 (含 0.01 非管理費用) 因此需要利用空格切割字串，存入第一部分，cost => ccost
ccost = []
for i in range(len(cost)):
    temp = str(cost[i]).split(' ')
    ccost.append(temp[0])
print(ccost)


db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()

for a in range(len(etf)):
    print(a)
    sql0 = "UPDATE detail SET etf全名 = '"+ name[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql0)
    sql1 = "UPDATE detail SET 創立時間 = '"+ create[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql1)
    sql2 = "UPDATE detail SET 投資區域 = '"+ area[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql2)
    sql3 = "UPDATE detail SET 投資標的 = '"+ buy_type[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql3)
    sql4 = "UPDATE detail SET 投資風格 = '"+ style[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql4)
    sql5 = "UPDATE detail SET 折溢價 = '"+ discount[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql5)
    sql6 = "UPDATE detail SET 殖利率 = '"+ interest[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql6)
    sql7 = "UPDATE detail SET 配息頻率 = '"+ interest_freq[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql7)
    sql8 = "UPDATE detail SET 管理費 = '"+ manage[a] +"' WHERE name = '" +etf[a]+"';"
    print(sql8)
    sql9 = "UPDATE detail SET 總費用率 = '"+ ccost[a] +"' WHERE name = '" +etf[a]+"';"
    print(sql9)
    sql10 = "UPDATE detail SET 年化標準差 = '"+ y_std[a] +"' WHERE name = '" +etf[a]+"';"
    print(sql10)
    sql11 = "UPDATE detail SET 追蹤指數 = '"+ trace_index[a] +"' WHERE name = '" +etf[a]+"'; "
    print(sql11)
    sql12 = "UPDATE detail SET 基準指數 = '"+ base_index[a] +"' WHERE name = '" +etf[a]+"';"
    print(sql12)

    
    try:
        cursor.execute(sql0)
        cursor.execute(sql1)
        cursor.execute(sql2)
        cursor.execute(sql3)
        cursor.execute(sql4)
        cursor.execute(sql5)
        cursor.execute(sql6)
        cursor.execute(sql7)
        cursor.execute(sql8)
        cursor.execute(sql9)
        cursor.execute(sql10)
        cursor.execute(sql11)
        cursor.execute(sql12)
        db.commit()
        print("Data are successfully inserted into detail")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)




############################################  etf_close 建置   ############################################

etf_stk = etf

db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()
today = datetime.date.today()


for xxx in range(len(etf_stk)):
    print(xxx+1)
    stk = yf.Ticker(etf_stk[xxx])
    # 取得 2000 年至今的資料
    data = stk.history(start='1990-01-01')
    # 簡化資料，只取開、高、低、收以及成交量
    data = data[['Close']]
    data['Date'] = data.index
    data.columns = ['close','date']
    data = data.reset_index()
    data = data.drop(['Date'],axis=1)


    found = data['date'][0]
    found_temp = str(found).split(' ')
    found_sql = found_temp[0]
    vary = relativedelta(today,found)
    found_limit = vary.years
    print(etf_stk[xxx])
    print(found_limit)

    length = data.shape[0]

    sql= "UPDATE detail SET `資料年限` ='%s',`資料起始` ='%s' WHERE `name` ='%s'" % (str(found_limit),str(found_sql),str(etf_stk[xxx]))

    cursor.execute(sql)
    db.commit()
    print("Data are successfully inserted into detail")


    # etf_close 欄位名稱
    createsqltable = """CREATE TABLE IF NOT EXISTS """ + 'etf_close '  + '(name VARCHAR(20),date date,close VARCHAR(100))'+  " DEFAULT CHARSET=utf8" + ";"
    print(createsqltable)
    cursor.execute(createsqltable)
    db.commit()
    print("etf_close table are successfully create")

    # 把input_data寫成sql語法
    for i in range(length):
        sql = "INSERT INTO etf_close (`name`,`date`,`close`) VALUES"
        values = "('%s','%s','%s')"
        sql += values % (etf_stk[xxx],data['date'][i],data['close'][i])
        cursor.execute(sql)
        db.commit()
    print("Data are successfully inserted into etf_close")

db.close()


############################################  close 建置   ############################################
# %%
db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()

sql = "CREATE TABLE IF NOT EXISTS close( date date," 

for i in range(len(etf_stk)):
    if i == (len(etf_stk)-1):
        sql = sql + etf_stk[i] + ' double)'
    else:
        sql = sql + etf_stk[i] + ' double,'
print(sql)
cursor.execute(sql)
db.commit()
print("close table are successfully create")
# %%
etf = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'VEA', 'IEFA', 'AGG', 'VWO', 'IEMG', 'GLD', 'BND', 'VUG', 'IWM', 'VTV', 'IWF', 'IJR', 'IJH', 'EFA', 'LQD', 'VIG', 'IWD', 'VCIT', 'VO', 'VGT', 'VB', 'VXUS', 'BNDX', 'XLK', 'VCSH', 'ITOT', 'VYM', 'VEU', 'IVW', 'USMV', 'IAU', 'VNQ', 'XLF', 'BSV', 'EEM', 'TIP', 'MBB', 'IXUS', 'IWB', 'SCHX', 'XLV', 'IWR', 'DIA', 'SCHF', 'HYG', 'ARKK', 'IGSB', 'VV', 'QUAL', 'MUB', 'VBR', 'MDY', 'PFF', 'IVE', 'RSP', 'SHY', 'XLY', 'EMB', 'SCHB', 'VT', 'SDY', 'SCHD', 'TLT', 'XLE', 'SHV', 'XLI', 'GDX', 'JPST', 'IWP', 'VBK', 'DVY', 'DGRO', 'BIV', 'VGK', 'VXF', 'ACWI', 'MTUM', 'SCHP', 'MINT', 'SCHA', 'GOVT', 'IEF', 'SLV', 'EWJ', 'VHT', 'VMBS', 'SCHG', 'IWN', 'ESGU', 'IWO', 'BIL', 'JNK', 'IWS', 'XLP', 'SCZ', 'VOE', 'GSLC', 'XLU', 'FDN', 'IEI', 'XLC', 'IGIB', 'IWV', 'IBB', 'VTEB', 'EFAV', 'VTIP', 'VLUE', 'ARKG', 'VOT', 'FVD', 'IUSG', 'EFG', 'VGSH', 'MGK', 'SPDW', 'SPYG', 'TQQQ', 'SCHE', 'IHI', 'SCHZ', 'EFV', 'SCHM', 'SCHV', 'IJK', 'FTCS', 'SPLV', 'EWY', 'SPLG', 'SPYV', 'VFH', 'IUSV', 'XBI', 'USHY', 'SCHO', 'IJS', 'SPSB', 'HYLB', 'OEF', 'NOBL', 'CWB', 'ESGE', 'PGX', 'MCHI', 'EWZ', 'VGIT', 'IYW', 'ICLN', 'BBJP', 'LMBS', 'XLB', 'AAXJ', 'IUSB', 'IJJ', 'ARKW', 'EWT', 'VSS', 'SPIB', 'IJT', 'GDXJ', 'SKYY', 'HDV', 'FPE', 'USIG', 'ACWV', 'SPEM', 'VCLT', 'SHYG', 'FNDX', 'FNDF', 'IGV', 'SPAB', 'BKLN', 'VDC', 'VONG', 'BLV', 'FLOT', 'SOXX', 'EZU', 'FTEC', 'DGRW', 'ICSH', 'INDA', 'VCR', 'FTSM', 'VNQI', 'IXN', 'VPL', 'VOOG', 'FIXD', 'AMLP', 'ISTB', 'TAN', 'IDEV', 'SCHH', 'NEAR', 'PRF', 'BBCA', 'SHM', 'VPU', 'SPTM', 'MOAT', 'FXI', 'VIS', 'EEMV', 'ANGL', 'SMH', 'ESGD', 'SUB', 'BOND', 'IYR', 'IEUR', 'VDE', 'FNDA', 'GLDM', 'XSOE', 'SCHR', 'IDV', 'ACWX', 'SJNK', 'FNDE', 'QLD', 'GUNR', 'DBEF', 'SPMD', 'VTWO', 'KWEB', 'MGV', 'ONEQ', 'SPSM', 'USO', 'VONV', 'EMLC', 'QTEC', 'TFI', 'IAGG', 'IWY', 'CIBR', 'MGC', 'BBEU', 'TOTL', 'XT', 'STIP', 'USSG', 'HYD', 'BBIN', 'IGM', 'VIGI', 'SLYV', 'ESGV', 'VOX', 'KBE', 'SSO', 'PDBC', 'SCHC', 'IGF', 'EWU', 'GSY', 'PBW', 'SPTS', 'EWC', 'JETS', 'VNLA', 'SUSL', 'AIA', 'QCLN', 'SPMB', 'SPTL', 'PCY', 'SRLN', 'IOO', 'ITA', 'RPG', 'DON', 'IQLT', 'LIT', 'SOXL', 'XOP', 'DSI', 'VWOB', 'FXL', 'EWG', 'DLN', 'SPHD', 'SGOL', 'VAW', 'KRE', 'IYH', 'BOTZ', 'FHLC', 'SPTI', 'IXJ', 'SPYD', 'ASHR', 'FAS', 'IGLB', 'GBIL', 'SPHQ', 'FV', 'ARKQ', 'REET', 'HEFA', 'RYT', 'SUSA', 'ARKF', 'FLRN', 'SPIP', 'BAB', 'GVI', 'VGLT', 'BSCM', 'FNDC', 'NFRA', 'GSIE', 'PZA', 'HYLS', 'SLYG', 'XLRE', 'MDYG', 'SLQD', 'EPP', 'FBT', 'HACK', 'RODM', 'BSCL', 'KOMP', 'RDVY', 'JHMM', 'PDP', 'FPX', 'VONE', 'GXC', 'HEDJ', 'GEM', 'ITB', 'HYS']

etf_stk = etf
stk = yf.Ticker(etf_stk[0])
# 取得 2000 年至今的資料
data = stk.history(start='1990-01-01')
# 簡化資料，只取開、高、低、收以及成交量
data = data[['Close']]

for xxx in range(1,len(etf_stk)):
    print(xxx)
    print(etf_stk[xxx])
    stk = yf.Ticker(etf_stk[xxx])
    # 取得 2000 年至今的資料
    data1 = stk.history(start='1990-01-01')
    # 簡化資料，只取開、高、低、收以及成交量
    data1 = data1[['Close']]
    data = pd.merge(data,data1, left_index=True, right_index=True, how='outer')

data.to_csv('D:/Alia/Documents/109-1/資產配置/database/close.csv', encoding='utf_8_sig')



db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()

# 先做出插入close的語法 
sql = "INSERT INTO close (`date`"
value = "("
for i in range(len(etf_stk)):
    sql = sql + ',`' + etf_stk[i] + '`'
    if i ==( len(etf_stk)-1):
        value = value + "'%s'" + ")"
    else:
        value = value + "'%s'" + ","
sql = sql+") VALUES"

# print(sql)
# print(value)

from csv import reader

with open('D:/Alia/Documents/109-1/資產配置/database/close.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    num = 0
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        # print(len(row))
        num += 1
        if num == 1 :
           continue
        temp = str(row)
        temp = temp.replace("'nan'", "NULL")
        temp = temp.replace("[", "(")
        temp = temp.replace("]", ")")
        temp = temp.replace("''", "NULL")
        sql_temp = sql
        sql_temp += temp
        cursor.execute(sql_temp)
        db.commit()

print("Data are successfully inserted into close")

db.close()


############################################  detail 配息率 建置   ############################################

# %%
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import pymysql
import datetime
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os

# %%
year = ["1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",
        "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020",
        "2021","2022","2023","2024","2025","2026","2027","2028","2029","2030"]


# 配息紀錄 欄位名稱
createsqltable_1 = """CREATE TABLE IF NOT EXISTS """ + '配息紀錄 '  + '(配息基準日 VARCHAR(20),除息日 VARCHAR(20),發放日 VARCHAR(20),配息總額 VARCHAR(20))'+  " DEFAULT CHARSET=utf8" + ";"
print(createsqltable_1)

# 配息率 欄位名稱
createsqltable_2 = """CREATE TABLE IF NOT EXISTS """ + '配息率 '  + '(date date,配息率 VARCHAR(50))'+  " DEFAULT CHARSET=utf8" + ";"
print(createsqltable_2)


# %%
avg_div = np.zeros(len(etf))
print(avg_div)
a=0
while a<len(etf) :
# while a<1 :
    print(etf[a])
    #配息爬蟲
    driver = webdriver.Chrome(ChromeDriverManager().install())
    url = "https://www.moneydj.com/ETF/X/Basic/Basic0005.xdjhtm?etfid="
    url += etf[a]
    print(url)
    driver.get(url)
    time.sleep(3)
    soup = bs(driver.page_source,"html.parser")
    raw_data = [data.text for data in soup.find_all("td",["col01","col02","col03","col07"])]
    
    sql = "INSERT INTO 配息紀錄 (`配息基準日`, `除息日`, `發放日`, `配息總額` ) VALUES"
    							
    values = "('%s','%s','%s','%s')"
    
    #print(len(raw_data))
    i = 0
    while(i < len(raw_data)):
        if (i < len(raw_data)-5):
            sql += values % (raw_data[i], raw_data[i + 1], raw_data[i + 2], raw_data[i + 3]) + ","
        else:
            sql += values % (raw_data[i], raw_data[i + 1], raw_data[i + 2], raw_data[i + 3]) 
        i += 4
    
    #print(sql)
    # print()
    driver.close()
    
    # 抓到的配息入資料庫
    db = pymysql.connect("localhost", "root", "esfortest", "test")
    cursor = db.cursor()
    try:
        cursor.execute(createsqltable_1)
        db.commit()
        print("配息紀錄 table are successfully create")
        cursor.execute(createsqltable_2)
        db.commit()
        print("配息率 table are successfully create")

        cursor.execute(sql)
        db.commit()
        print("Data are successfully inserted")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)
    #配息爬蟲結束
    
    
    
    #計算每次的配息率
    sql="SELECT * FROM `配息紀錄`"
    cursor.execute(sql)
    result_select_div = cursor.fetchall()
    db.commit()
    
    sql3 = "INSERT INTO 配息率 (`date`, `配息率` ) VALUES"
    values = "('%s','%s')"
    
    i=0
    k=1
    #每筆找相對應日期的市價並相除
    while i<len(result_select_div):
        sql2 = "select * from etf_close where (name = '"+etf[a]+"' and date = '"#從yahoo匯入的找市價
        sql2 += (str(result_select_div[i][k]) +"')")
        print(sql2)
        cursor.execute(sql2)
        result_select_close = cursor.fetchall()
        db.commit()
        print(result_select_close)
        if len(result_select_close)==0:
            dividend = 0
            close = 0
        else:
            close = float(result_select_close[0][2])
            dividend = float(result_select_div[i][3])
        # print(result_select_div[i][k])
        print(close)
        # print(dividend)
        if close!=0:
            div_percent = dividend/close
        else:
            div_percent = 0
        # print(div_percent)
        if(i<len(result_select_div)-1):
            sql3 += values % (result_select_div[i][k], div_percent) +","
        else:
            sql3 += values % (result_select_div[i][k], div_percent)
        #print(sql3)
        i+=1
    
    print(sql3)
    
    #每次配息率入資料庫
    try:
        cursor.execute(sql3)
        db.commit()
        print("Div_percent are successfully inserted")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)
    #db.close
    #計算每次配息率結束
    
    
    #計算年平均配息率
    now_year = 2019
    now_year -= 1990
    i = 0
    dividend = np.empty(now_year+1)
    freq = np.empty(now_year+1)
    
    #每年分別抓出來算總和與配息次數
    while i<=now_year :
        sql = "select * from 配息率 where date like '" + year[i]+"%'"
        cursor.execute(sql)
        result_select = cursor.fetchall()
        db.commit()
        # print(i)
        # print(result_select)
        dividend[i] = 0
        j=0
        freq[i] = len(result_select)
        while j<len(result_select) :
            dividend[i] += float(result_select[j][1])
            j+=1
        # print(dividend[i])
        # print(freq[i])
        i+=1
    #db.close()
    
    # print(dividend)
    # print(freq)
    
    #算平均
    count = 0
    total = 0.0 
    i=0
    freq_year = freq[now_year-1]
    # print(freq_year)
    # print(dividend[i] *freq_year /freq[i])
    while i < len(dividend):
        #total += (dividend[i] *freq_year /freq[i])
        if(dividend[i]!=0):
            total += (dividend[i] *freq_year /freq[i])
            count+=1
        i+=1
    # print(total)
    # print(count)
    avg_dividend = total/count
    print(avg_dividend)
    avg_div[a] = avg_dividend
    
    #計算年平均配息率結束
    
    
    #清空資料庫
    #db = pymysql.connect("localhost", "root", "esfortest", "etf")
    #cursor = db.cursor()
    
    sql_del1 = "TRUNCATE TABLE `配息率`"
    try:
        cursor.execute(sql_del1)
        db.commit()
        print("delete1 successful")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)
    sql_del2 = "TRUNCATE TABLE `配息紀錄`"
    try:
        cursor.execute(sql_del2)
        db.commit()
        print("delete2 successful")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)
    db.close()
    print(etf[a]+' end')
    print()
    a+=1

print(avg_div)
# %%
#[0.0168106  0.02033887 0.04343551 0.01860585 0.031083   0.01811477 0.01843766 0.0444699  0.02534931 0.01579987 0.0122712  0.02636138
# 0.05635492 0.0276834  0.00949157 0.02104849 0.03353616 0.02008875 0.0364111  0.03023534 0.01915196 0.03689285 0.02206909 0.03907675 0.01410118]
# [0.02124299 0.01981818 0.04267185 0.01837447 0.03090534 0.01811477
#  0.01830685 0.0439066  0.02509347 0.01211015 0.02612884 0.05580605
#  0.02719655 0.00937193 0.02090058 0.03335623 0.019201   0.03604143
#  0.03013608 0.01915196 0.03689285 0.02206909 0.03860267 0.01389153]
db = pymysql.connect("localhost", "root", "esfortest", "test")
cursor = db.cursor()
i=0
while i<len(etf):
    sql = 'UPDATE detail SET 配息率 = '+ str(avg_div[i]) +" WHERE name = '"+etf[i]+"'"
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
        print("Data are successfully update")
    except Exception as e:
        db.rollback()
        print("Exception Occured : ", e)
    i+=1
db.close()

