# %%
import pymysql
from dateutil.relativedelta import relativedelta
import datetime
import yfinance as yf
import pandas as pd
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
# %%
############################################  volume 建置   ############################################
db_name = 'my_etf'
db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
cursor = db.cursor()

list_etf = ['TWO_etf']
tmp1 = get_etfs(db_name,list_etf)
list_etf = ['TW_etf']
tmp2 = get_etfs(db_name,list_etf)
list_etf = ['US_etf']
tmp3 = get_etfs(db_name,list_etf)
etf = tmp3+tmp2+tmp1
etf = tmp1+tmp2+tmp3
print(etf)

# %%

etf_stk = etf
# stk = yf.Ticker(etf_stk[0])
# # 取得 2000 年至今的資料
# data = stk.history(start='1990-01-01')
# # 簡化資料，只取開、高、低、收以及成交量
# data = data[['Volume']]

# for xxx in range(1,len(etf_stk)):
#     print(xxx,etf_stk[xxx])
#     if xxx < (len(tmp3)+len(tmp2)):
#         stk = yf.Ticker(etf_stk[xxx])
#     else:
#         stk = yf.Ticker(etf_stk[xxx]+'O')
#     # 取得 2000 年至今的資料
#     data1 = stk.history(start='1990-01-01')
#     # 簡化資料，只取開、高、低、收以及成交量
#     data1 = data1[['Volume']]
#     data = pd.merge(data,data1, left_index=True, right_index=True, how='outer')

# data.to_csv('D:/Alia/Documents/109-1/資產配置/database/volume.csv', encoding='utf_8_sig')

# %%
# sql = "CREATE TABLE IF NOT EXISTS volume ( date date," 

# for i in range(len(etf_stk)):
#     if i == (len(etf_stk)-1):
#         sql = sql + '`' + etf_stk[i] +'` double)'
#     else:
#         sql = sql +'`' + etf_stk[i] +'` double,'
# print(sql)
# cursor.execute(sql)
# db.commit()
# print("volume table are successfully create")

# 先做出插入volume的語法 
sql = "INSERT INTO volume (`date`"
value = "("
for i in range(len(etf_stk)):
    sql = sql + ',`' + etf_stk[i] + '`'
    if i ==( len(etf_stk)-1):
        value = value + "'%s'" + ")"
    else:
        value = value + "'%s'" + ","
sql = sql+") VALUES"

print(sql)
# print(value)

from csv import reader

with open('D:/Alia/Documents/109-1/資產配置/database/volume.csv', 'r') as read_obj:
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

print("Data are successfully inserted into volume")

db.close()

