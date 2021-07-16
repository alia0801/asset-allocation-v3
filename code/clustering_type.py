# %%
from operator import index, le
from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import pymysql
import time
import datetime
# %%
#<--------------------------------------------物件準備(新)-------------------------------------------->#

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

def cluser_by_type(list_etf,db_name='my_etf'):
    # db_name = 'my_etf'
    # list_etf = ['TW_etf']
    etfs = get_etfs(db_name,list_etf)
    types = []
    df_list = []
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()

    for etf in etfs:
        sql = "select 投資標的 from `detail` where name = '"+etf+"'"
        cursor.execute(sql)
        result_select = cursor.fetchall()
        db.commit()
        if result_select[0][0] in types:
            etf_type = types.index(result_select[0][0])
        else:
            etf_type = len(types)
            types.append(result_select[0][0])
        df_list.append([etf,etf_type+1])
    df = pd.DataFrame(df_list,columns=['etf','type'])
    # print(types)
    # print(df)
    return types,df
# %%
if __name__ == '__main__':
    db_name = 'my_etf'
    list_etf = ['TW_etf']
    types,ans_df = cluser_by_type(list_etf,db_name)



# %%
