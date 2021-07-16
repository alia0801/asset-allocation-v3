# %%
import pymysql
from dateutil.relativedelta import relativedelta
import datetime
import yfinance as yf
# %%
# 根據type_name查詢record，取出對應的code的list回傳
def get_etfs(db,type_name_list):
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db))
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

db_name = 'my_etf'
# list_etf = ['TW_etf']
# list_etf = ['TWO_etf']
list_etf = ['US_etf']
etfs = get_etfs(db_name,list_etf)
db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
cursor = db.cursor()
today = datetime.date.today()
# print(etfs[722])
# stk = yf.Ticker(etfs[722])

# %%
for xxx in range(722,len(etfs)):
    print(xxx+1)
    stk = yf.Ticker(etfs[xxx])
    # 取得 2000 年至今的資料
    data = stk.history(start='1990-01-01')
    # 簡化資料，只取開、高、低、收以及成交量
    data = data[['Volume']]
    data['Date'] = data.index
    data.columns = ['volume','date']
    data = data.reset_index()
    data = data.drop(['Date'],axis=1)
    # print(data)
    # break


    # found = data['date'][0]
    # found_temp = str(found).split(' ')
    # found_sql = found_temp[0]
    # vary = relativedelta(today,found)
    # found_limit = vary.years
    # print(etfs[xxx])
    # print(found_limit)

    length = data.shape[0]

    # sql= "UPDATE detail SET `資料年限` ='%s',`資料起始` ='%s' WHERE `name` ='%s'" % (str(found_limit),str(found_sql),str(etfs[xxx]))

    # cursor.execute(sql)
    # db.commit()
    # print("Data are successfully inserted into detail")


    # etf_close 欄位名稱
    createsqltable = """CREATE TABLE IF NOT EXISTS """ + 'etf_volume '  + '(name VARCHAR(20),date date,volume VARCHAR(100))'+  " DEFAULT CHARSET=utf8" + ";"
    print(createsqltable)
    cursor.execute(createsqltable)
    db.commit()
    print("etf_volume table are successfully create")

    # 把input_data寫成sql語法
    for i in range(length):
        sql = "INSERT INTO etf_volume (`name`,`date`,`volume`) VALUES"
        values = "('%s','%s','%s')"
        sql += values % (etfs[xxx],data['date'][i],data['volume'][i])
        cursor.execute(sql)
        db.commit()
    print("Data are successfully inserted into etf_volume")
    # break

db.close()


