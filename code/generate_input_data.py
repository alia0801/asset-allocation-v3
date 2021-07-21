# %%
from datetime import date, datetime
import pymysql
import clustering_corr
import clustering_type
import pandas as pd
import math
import statistics
# %%
def generate_group(cluster,y,month,nnnn,number,market_etf,db_name,list_etf):
    if cluster=='type':
        types,ans_df = clustering_type.cluser_by_type(list_etf,db_name)
        number = len(types)


    elif cluster=='corr':
        # (y,expect_reward,nnnn) = (2018,0.08,3)
        # number=3
        # import clustering_corr
        med_id,med_name,ans_df = clustering_corr.cluser_by_corr(list_etf,db_name,number,y,month,nnnn)

    groups=[]
    groups.append([market_etf])
    for i in range(number):
        groups.append([])
    for i in range(len(ans_df)):
        t = ans_df['type'][i]
        groups[t].append(ans_df['etf'][i])
    # print(groups)
    return groups,number+1

# %%
def df_avg(df):
    df['avg']=0
    pre_avg = 0
    for i in range(len(df)):
        avg = 0
        count=0
        for j in range(1,len(df.T)-1):
            if str(type(df[j][i]))!="<class 'NoneType'>" and str(df[j][i])!='nan':
                # print(str(df[j][i]))
                avg+=df[j][i]
                count += 1
        # print(avg)
        if count!=0:
            avg = avg/count
        else:
            avg = pre_avg
        df.loc[i,'avg'] = avg
        pre_avg = avg
        # break
    df = df.rename(columns={0:'date'})
    df = df[['date','avg']]
    return df

# def cal_log(row):    
#     row['log'] = math.log((row['avg']))
#     return row

def df_avg2hv(df,ans_len):
    df['log'] = 0
    for i in range(len(df)):
        if df['avg'][i]!=0:
            df.loc[i,'log'] = math.log(df['avg'][i])
        else:
            df.loc[i,'log'] = -1
    # df = df.apply(cal_log, axis=1)
    df['chg']=0
    
    for i in range(len(df)-1):
        df.loc[i+1,'chg'] = df['log'][i+1] - df['log'][i]

    l = len(df)-ans_len
    volatility = []
    for i in range(ans_len):
        # df.loc[i+31,'hv'] = statistics.pstdev(df['chg'][i+1:31+i])
        v = statistics.pstdev(df['chg'][i:i+l]) * math.sqrt(252)
        volatility.append(v)


    # volatility = statistics.pstdev(df['chg'])
    return volatility

def generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf):
    groups,number = generate_group(cluster,y,month,nnnn,number,market_etf,db_name,list_etf)

    date1 = date(y,month,1)
    if month==12:
        date2 = date(y+1,1,1)
    else:
        date2 = date(y,month+1,1)
    if month==1:       
        date3 = date(y-1,12,1)
    else:
        date3 = date(y,month-1,1)    
    

    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()

    closes = []
    volumes = []
    volatilitys = []
    every_close_finalday=[]
    for i in range(number):
        tmp = []
        group = groups[i]
        sql = 'select `date`,'
        for j in range(len(group)):
            etf = group[j]
            if j!=len(group)-1:
                sql += ('`'+etf+'`,')
            else:
                sql += ('`'+etf+'`')
        # SELECT * from (SELECT date,SPY FROM close WHERE date < '2020-03-01' order by date DESC limit 21) AS C order by C.date
        # sql_close  = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date2)+"' order by date DESC limit 21) AS C order by C.date")
        # sql_close2 = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date2)+"' order by date DESC limit 42) AS C order by C.date")
        # sql_volume = "SELECT * from (" + sql + (" from `volume` where date < '"+str(date2)+"' order by date DESC limit 21) AS C order by C.date")
        sql_close  = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date1)+"' order by date DESC limit 21) AS C order by C.date")
        sql_close2 = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date1)+"' order by date DESC limit 42) AS C order by C.date")
        sql_volume = "SELECT * from (" + sql + (" from `volume` where date < '"+str(date1)+"' order by date DESC limit 21) AS C order by C.date")
       # print(sql_close)
        # print(sql_volume)
        cursor.execute(sql_close)
        result_close = cursor.fetchall()
        cursor.execute(sql_close2)
        result_close2 = cursor.fetchall()
        cursor.execute(sql_volume)
        result_volume = cursor.fetchall()
        
        df_close = pd.DataFrame(list(result_close))
        df_close2 = pd.DataFrame(list(result_close2))
        df_volume = pd.DataFrame(list(result_volume))

        for j in range(len(group)):
            # tmp.append(df_close[j][len(df_close)-1])
            etf = group[j]
            sql_scale = "SELECT 規模 FROM `detail` WHERE name = '"+ etf +"'"
            cursor.execute(sql_scale)
            result_scale = cursor.fetchall()
            scale = result_scale[0][0]
            tmp.append(float((scale.split('(')[0]).replace(',','')))
        
        df_close_avg = df_avg(df_close)
        df_close_avg2 = df_avg(df_close2)
        df_volume_avg = df_avg(df_volume)
        # if df_close_avg['avg'].sum==0:
        #     vol = 0
        # else:
        #     vol = df_avg2hv(df_close_avg)
        vol = df_avg2hv(df_close_avg2,len(df_close_avg))
        
        closes.append(list(df_close_avg['avg']))
        volumes.append(list(df_volume_avg['avg']))
        volatilitys.append(vol)
        every_close_finalday.append(tmp)
        # break
    return closes,volumes,volatilitys,groups,number,every_close_finalday

# %%
def generate_training_data(y,month,db_name,groups,number):
    # groups,number = generate_group(cluster,y,expect_reward,nnnn,number,market_etf)

    date1 = date(y,month,1)
    # date2 = date(y,month+1,1)
    # date3 = date(y,month-1,1)

    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()

    closes = []
    volumes = []
    volatilitys = []
    for i in range(number):
        group = groups[i]
        sql = 'select `date`,'
        for j in range(len(group)):
            etf = group[j]
            if j!=len(group)-1:
                sql += ('`'+etf+'`,')
            else:
                sql += ('`'+etf+'`')
        # SELECT * from (SELECT date,SPY FROM close WHERE date < '2020-03-01' order by date DESC limit 21) AS C order by C.date
        sql_close  = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date1)+"' order by date DESC limit 252) AS C order by C.date")
        sql_close2 = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date1)+"' order by date DESC limit 273) AS C order by C.date")
        sql_volume = "SELECT * from (" + sql + (" from `volume` where date < '"+str(date1)+"' order by date DESC limit 252) AS C order by C.date")
        # print(sql_close)
        # print(sql_volume)
        cursor.execute(sql_close)
        result_close = cursor.fetchall()
        cursor.execute(sql_close2)
        result_close2 = cursor.fetchall()
        cursor.execute(sql_volume)
        result_volume = cursor.fetchall()
        
        df_close = pd.DataFrame(list(result_close))
        df_close2 = pd.DataFrame(list(result_close2))
        df_volume = pd.DataFrame(list(result_volume))
        
        df_close_avg = df_avg(df_close)
        df_close_avg2 = df_avg(df_close2)
        df_volume_avg = df_avg(df_volume)
        # if df_close_avg['avg'].sum==0:
        #     vol = 0
        # else:
        #     vol = df_avg2hv(df_close_avg)
        vol = df_avg2hv(df_close_avg2,len(df_close_avg))
        
        closes.append(list(df_close_avg['avg']))
        volumes.append(list(df_volume_avg['avg']))
        volatilitys.append(vol)
        # break
    return closes,volumes,volatilitys

# %%


def generate_data_d(y,nnnn,month,db_name,cluster,number,market_etf,list_etf,groups):
    # groups,number = generate_group(cluster,y,expect_reward,nnnn,number,market_etf,db_name,list_etf)

    date1 = date(y,month,1)
    if month==12:
        date2 = date(y+1,1,1)
    else:
        date2 = date(y,month+1,1)
    if month==1:       
        date3 = date(y-1,12,1)
    else:
        date3 = date(y,month-1,1)    
    

    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()

    closes = []
    volumes = []
    volatilitys = []
    every_close_finalday=[]
    for i in range(number):
        tmp = []
        group = groups[i]
        sql = 'select `date`,'
        for j in range(len(group)):
            etf = group[j]
            if j!=len(group)-1:
                sql += ('`'+etf+'`,')
            else:
                sql += ('`'+etf+'`')
        # SELECT * from (SELECT date,SPY FROM close WHERE date < '2020-03-01' order by date DESC limit 21) AS C order by C.date
        sql_close  = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date2)+"' order by date DESC limit 21) AS C order by C.date")
        sql_close2 = "SELECT * from (" + sql + (" from `close`  where date < '"+str(date2)+"' order by date DESC limit 42) AS C order by C.date")
        sql_volume = "SELECT * from (" + sql + (" from `volume` where date < '"+str(date2)+"' order by date DESC limit 21) AS C order by C.date")
        # print(sql_close)
        # print(sql_volume)
        cursor.execute(sql_close)
        result_close = cursor.fetchall()
        cursor.execute(sql_close2)
        result_close2 = cursor.fetchall()
        cursor.execute(sql_volume)
        result_volume = cursor.fetchall()
        
        df_close = pd.DataFrame(list(result_close))
        df_close2 = pd.DataFrame(list(result_close2))
        df_volume = pd.DataFrame(list(result_volume))

        for j in range(len(group)):
            # tmp.append(df_close[j][len(df_close)-1])
            etf = group[j]
            sql_scale = "SELECT 規模 FROM `detail` WHERE name = '"+ etf +"'"
            cursor.execute(sql_scale)
            result_scale = cursor.fetchall()
            scale = result_scale[0][0]
            tmp.append(float((scale.split('(')[0]).replace(',','')))
        
        df_close_avg = df_avg(df_close)
        df_close_avg2 = df_avg(df_close2)
        df_volume_avg = df_avg(df_volume)
        # if df_close_avg['avg'].sum==0:
        #     vol = 0
        # else:
        #     vol = df_avg2hv(df_close_avg)
        vol = df_avg2hv(df_close_avg2,len(df_close_avg))
        
        closes.append(list(df_close_avg['avg']))
        volumes.append(list(df_volume_avg['avg']))
        volatilitys.append(vol)
        every_close_finalday.append(tmp)
        # break
    return closes,volumes,volatilitys,groups,number,every_close_finalday

# %%
if __name__ == '__main__':
    db_name = 'my_etf'
    list_etf = ['TW_etf']
    # list_etf = ['US_etf']
    (y,nnnn,month) = (2018,1,6) # 2018/1/1~2018/12/31
    market_etf = '006204.TW'
    # market_etf = 'SPY'

    number = 3
    # cluster = 'type'
    cluster = 'corr'
    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    train_closes,train_volumes,train_volatilitys = generate_training_data(y,month,db_name,groups,number)

    # df_close = pd.DataFrame(closes)
    # df_volumes = pd.DataFrame(volumes)
    # df_vol = pd.DataFrame(volatilitys)

    # print(df_close)
    # print(df_volumes)
    # print(df_vol)

    # datas=[]
    # for i in range(number):
    #     c = closes[i]
    #     v = volumes[i]
    #     vol = volatilitys[i]
    #     tmp = [c,v,vol]
    #     datas.append(tmp)
    #     print(tmp)



    

# %%
