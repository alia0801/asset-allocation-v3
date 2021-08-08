# %%
import os
os.environ['PYTHONHOME'] = r'D:\Alia\Anaconda3'
os.environ['PYTHONPATH'] = r'D:\Alia\Anaconda3\Lib\site-packages'
os.environ['R_HOME'] = r'C:\Program Files\R\R-3.6.1'
os.environ['R_USER'] = r'D:\Alia\Anaconda3\Lib\site-packages\rpy2' #path depe
import sys
import pandas as pd
import numpy as np
import pymysql
import math
import statistics
import time
import datetime
# from itertools import combinations, permutations
# from scipy.special import comb, perm
import numpy_financial as npf
from dateutil.relativedelta import relativedelta
import csv
import operator
# %%
#<--------------------------------------------物件準備(新)-------------------------------------------->#
# ETF的物件製作
def do_etf_obj(etf,df_reward_std):
    class Etfs:
        def _init_(self,the_name,the_code,the_reward,the_risk):
            self.the_name = the_name
            self.the_code = the_code
            self.the_reward = the_reward
            self.the_risk = the_risk

    etf_target = []
    for i in range(df_reward_std.shape[0]):
        mazda = Etfs()
        mazda.the_name = str(df_reward_std['2'][i])
        mazda.the_code = etf.index(str(df_reward_std['2'][i]))
        mazda.the_reward = float(df_reward_std['3'][i])
        mazda.the_risk = float(df_reward_std['4'][i])
        etf_target.append(mazda)

    return etf_target


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

# 將各長年化值 以及相關性矩陣拿出來(可取出對應的code)
def get_ann_data_corr(db_name,today,nnnn,etf):
    df_reward_std = pd.DataFrame(columns=['2', '3', '4'])
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()
    for i in range(len(etf)):
        sql = "select * from `各長年化值` where (year =  '"+str(today.year) +"' and length = '"+str(nnnn) +"' and name = '"+str(etf[i]) +"')"
        cursor.execute(sql)
        result_select = cursor.fetchall()
        # print(result_select)
        db.commit()
        if (len(result_select))!=0:
            df_reward_std.loc[i] = [result_select[0][2], result_select[0][3], result_select[0][4]]

    df_reward_std = df_reward_std.reset_index(drop=True)

    # 這裡把nnnn年內的收盤價的相關性矩陣計算出來
    input_etf = []
    for i in range(len(df_reward_std)):
        input_etf.append('`' + str(df_reward_std['2'][i]) + '`')
    input_etf = ",".join(input_etf)  
    seefarday = datetime.date(int(today.year)-nnnn,today.month,today.day)
    constrain = " where date >= " + "'" +str(seefarday)+ "'" + " AND date <= " + "'"+str(today)+ "'"
    sql = "select " + 'date,' +input_etf +  " from close"
    sql = sql + constrain + " ORDER BY `close`.`date` DESC"
    cursor.execute(sql)
    result_select = cursor.fetchall()
    db.commit()
    df = pd.DataFrame(list(result_select))
    df = df.drop([0],axis=1)
    corr_pd1 = df.corr()
    return df_reward_std,corr_pd1

def cluser_by_corr(list_etf,db_name,number,y,month,nnnn=1):

    etf = get_etfs(db_name,list_etf)
    real_today = datetime.date.today()
    # today = datetime.date(y-1,12,31)
    today = datetime.date(y,month,1) # date end
    if (real_today-today).days > 0:
        df_reward_std,corr_pd1 = get_ann_data_corr(db_name,today,nnnn,etf)
        etf_target = do_etf_obj(etf,df_reward_std)
        for_clustering = corr_pd1.copy()
        for_clustering.columns = df_reward_std['2']
        for_clustering.index = df_reward_std['2']
        # for_clustering = (for_clustering-for_clustering.mean())/(for_clustering.std())
        # for_clustering = (for_clustering-for_clustering.min())/(for_clustering.max()-for_clustering.min())
        for_clustering = for_clustering.fillna(value=0)
        for_clustering.to_csv('my_data.csv', index=True)
        import rpy2.robjects as robjects
        robjects.r.source('txo.R')
        # number = 3
        condition = 'test(' + str(number) +  ')'
        a = robjects.r(condition)
        f = open('med.txt')
        i = 0 
        for line in f.readlines():
            if i == 0: 
                med_id = line
            else:
                med_name = line
            i = i + 1
        f.close()
        med_id = med_id.split('\n')
        med_id = med_id[0].split(' ')
        med_name = med_name.split('\n')
        med_name = med_name[0].split(' ')
        df = pd.read_csv('clustering.csv',index_col = 0)
        # print(med_id)
        # print(med_name)
        # print(df)
        df = df.reset_index()
        df = df.rename(columns = {'index': 'etf', 'x': 'type'}, inplace = False)
        return med_id,med_name,df,number
    else:
        return 0,0,0

# %%
if __name__ == '__main__':
    db_name = 'my_etf'
    list_etf = ['TW_etf']
    (y,month,nnnn) = (2018,6,1)
    number=3
    med_id,med_name,df,number = cluser_by_corr(list_etf,db_name,number,y,month,nnnn)
    print(med_id)
    print(med_name)
    print(df)

# %%
