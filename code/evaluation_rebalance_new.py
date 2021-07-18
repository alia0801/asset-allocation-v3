# %%
import pandas as pd
import numpy as np
import pymysql
import datetime
import math
import sys
import statistics
from dateutil.relativedelta import relativedelta

# %%
# db_name = 'tw_etf'
# db_name = 'test'
db_name = 'my_etf'


today = datetime.date.today()
# db = pymysql.connect("localhost", "root", "esfortest", db_name)
db = pymysql.connect(host="localhost", user="root", password="esfortest", database=db_name)

cursor = db.cursor()
sql = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.Columns Where Table_Name = 'close' AND TABLE_SCHEMA = '"+ db_name +"' AND ORDINAL_POSITION>1 ORDER BY `Columns`.`ORDINAL_POSITION` ASC"
cursor.execute(sql)
result_select = cursor.fetchall()
db.commit()
# print(list(result_select))
etf = []
for i in range(len(result_select)):
    etf.append(result_select[i][0])
# print(etf)
v1 = etf
# %%
# ans_df = pd.read_csv('D:/Alia/Documents/109-1/資產配置/train_test/train_3year_risk.csv')
# ans_df

# %%

sql = 'select * FROM `close` where date > 2004-12-31'
# print(sql)
cursor.execute(sql)
result_select = cursor.fetchall()
close_df = pd.DataFrame(list(result_select))
close_df = close_df.rename(columns={0:'date'})
for a in range(len(v1)):
    close_df = close_df.rename(columns={a+1:v1[a]})

# %%

# nnnn=1
# input_month_total=10000
# first_input_total=150000
day_of_month = [ 31,28,31, 30,31,30, 31,31,30, 31,30,31]

def get_a_etf(reward,y,length):
    sql = "SELECT * FROM `各長年化值` WHERE (reward>"+str(round(reward))+" and year="+ str(y) +" and length="+ str(length)+") ORDER BY `各長年化值`.`std` ASC"
    print(sql)
    cursor.execute(sql)
    result_select = cursor.fetchall()
    if len(result_select)>0:
        return result_select[0]
    else:
        return None    


def get_rewards(y,nnnn,choose,weight,first_input_total,input_month_total,mode=3):
    # print(choose,weight)
    if y+nnnn>today.year:
        print('y+nnnn too big')
        return [-100,-100,100,100,[],[]] #[reward,annual_reward,v1_std,mdd,sum_money]

    start_date = datetime.date(y,1,1)
    final_date = datetime.date(y+nnnn-1,12,31)
    
    while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
        start_date += datetime.timedelta(days=1)
    # print(start_date)
    start_date_index = list(close_df['date']).index(start_date)
    while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
        final_date -= datetime.timedelta(days=1)
    # print(start_date,final_date)
    final_date_index = list(close_df['date']).index(final_date)


    df = close_df[start_date_index:final_date_index+1]
    df = df.reset_index(drop=True)
    tmp=['date']
    for i in range(len(choose)):
        tmp.append(choose[i])
    # print(tmp)

    df = df[tmp]
    # print(df)
    
    for i in range(len(df)):
        have_nan = True
        for a in range(len(choose)):
            if np.isnan(df[choose[a]][i])==True: #是空的
                have_nan = False
                break
        if have_nan==False:
            # print(df['date'][i])
            df = df.drop([i],axis=0)          
    df2 = df.reset_index(drop=True)

    if len(df2)==0:
        # print(df2)
        print('not create')
        return [-100,-100,100,100,[],[]]

    sum_money,ratios = money_sim(choose,weight,df2,start_date,mode,first_input_total,input_month_total)
    # print(sum_money)

    input_money = [first_input_total]
    sum_money_yyy=[sum_money[0]]
    pre_y = df2['date'][0].year
    for i in range(1,len(sum_money)):
        yyy = df2['date'][i].year
        ttt = input_money[-1]
        if yyy==pre_y:
            # input_money.append(ttt)
            # df = df.drop([i],axis=0)
            pass
        else:
            sum_money_yyy.append(sum_money[i])
            input_money.append(ttt+input_month_total*12)
        pre_y = yyy   
    sum_money_yyy.append(sum_money[-1])
    input_money.append(input_money[-1]+input_month_total*12)
    # print(sum_money_yyy)
    # print(input_money)

    ann_reward=[]
    for i in range(1,len(sum_money_yyy)):
        r = (sum_money_yyy[i]-input_money[i])/input_money[i]
        ann = math.pow( (1+r), 1/i )-1
        ann_reward.append(ann)
    # print(ann_reward)   

    return ann_reward


    # reward = (sum_money[-1]-total_input_money)/total_input_money
    # print('reward',reward)
    # annual_reward = math.pow( (1+reward), 1/nnnn )-1
    # print('annual reward',annual_reward)

# %%

def train_choose(y,month,choose,weight,first_input_total,input_month_total,mode=3):
    print(choose,weight)
    nnnn = 1
    if y>today.year:
        print('y too big')
        return [-100,-100,100,100,[],[]] #[reward,annual_reward,v1_std,mdd,sum_money]

    start_date = datetime.date(y-1,month,1)
    final_date = datetime.date(y,month,1)
    
    while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
        start_date += datetime.timedelta(days=1)
    # print(start_date)
    start_date_index = list(close_df['date']).index(start_date)
    while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
        final_date -= datetime.timedelta(days=1)
    print(start_date,final_date)
    final_date_index = list(close_df['date']).index(final_date)


    df = close_df[start_date_index:final_date_index+1]
    df = df.reset_index(drop=True)
    tmp=['date']
    for i in range(len(choose)):
        tmp.append(choose[i])

    df = df[tmp]
    # print(df)
    
    for i in range(len(df)):
        have_nan = True
        for a in range(len(choose)):
            if np.isnan(df[choose[a]][i])==True: #是空的
                have_nan = False
                break
        if have_nan==False:
            # print(df['date'][i])
            df = df.drop([i],axis=0)          
    df2 = df.reset_index(drop=True)

    if len(df2)==0:
        # print(df2)
        print('not create')
        return [-100,-100,100,100,[],[]]

    sum_money,ratios = money_sim(choose,weight,df2,start_date,mode,first_input_total,input_month_total)
    # print(sum_money)
    # print(len(ratios))
    # print(len(df2))

    input_money = [first_input_total]
    sum_money_mmm=[sum_money[0]]
    pre_m = df2['date'][0].month
    for i in range(1,len(sum_money)):
        m = df2['date'][i].month
        ttt = input_money[-1]
        if m==pre_m:
            input_money.append(ttt)
        else:
            sum_money_mmm.append(sum_money[i])
            input_money.append(ttt+input_month_total)
        pre_m = m   
    sum_money_mmm.append(sum_money[-1])
    print(sum_money_mmm)
    
    total_input_money = first_input_total+input_month_total*(len(sum_money_mmm)-1)
    # print(sum_money[-1],total_input_money)
    reward = (sum_money[-1]-total_input_money)/total_input_money
    print('reward',reward)
    # ans_reward.append(reward)
    annual_reward = math.pow( (1+reward), 1/nnnn )-1
    print('annual reward',annual_reward)
    # ans_ann_reward.append(annual_reward)

    m_rews = []

    for i in range(len(sum_money_mmm)):
        if i==0:
            input = first_input_total
        else:
            input = sum_money_mmm[i-1]
        # print(sum_money[i],input)
        rrr = (sum_money_mmm[i]-input)/input
        m_rews.append(rrr)

    # print(m_rews)
    std_dev_m = statistics.stdev(m_rews)
    v1_std = std_dev_m * math.pow( 12, 0.5 )
    print('std_dev',v1_std)
    # ans_dev.append(v1_std)

    mdd = cal_mdd(choose,weight,df2,ratios)
    print('mdd',mdd)
    # mdd = 0 

    return [reward,annual_reward,v1_std,mdd,sum_money,input_money] # ,success_ratio,success

# %%
def test_choose(y,month,choose,weight,first_input_total,true_first_in,input_month_total,mode=3):
    print(choose,weight)
    nnnn = 1/12
    if y+nnnn>today.year:
        print('y+nnnn too big')
        return [-100,-100,100,100,[],[]] #[reward,annual_reward,v1_std,mdd,sum_money]

    start_date = datetime.date(y,month,1)
    if month==12:
        final_date = datetime.date(y+1,1,1)
    else:
        final_date = datetime.date(y,month+1,1)
    
    while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
        start_date += datetime.timedelta(days=1)
    # print(start_date)
    start_date_index = list(close_df['date']).index(start_date)
    while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
        final_date -= datetime.timedelta(days=1)
    print(start_date,final_date)
    final_date_index = list(close_df['date']).index(final_date)


    df = close_df[start_date_index:final_date_index+1]
    df = df.reset_index(drop=True)
    tmp=['date']
    for i in range(len(choose)):
        tmp.append(choose[i])

    df = df[tmp]
    # print(df)
    
    for i in range(len(df)):
        have_nan = True
        for a in range(len(choose)):
            if np.isnan(df[choose[a]][i])==True: #是空的
                have_nan = False
                break
        if have_nan==False:
            # print(df['date'][i])
            df = df.drop([i],axis=0)          
    df2 = df.reset_index(drop=True)

    if len(df2)==0:
        # print(df2)
        print('not create')
        return [-100,-100,100,100,[],[]]

    sum_money,ratios = money_sim(choose,weight,df2,start_date,mode,first_input_total,input_month_total)
    # print(sum_money)
    # print(len(ratios))
    # print(len(df2))

    input_money = [true_first_in]
    sum_money_mmm=[sum_money[0]]
    pre_m = df2['date'][0].month
    for i in range(1,len(sum_money)):
        m = df2['date'][i].month
        ttt = input_money[-1]
        if m==pre_m:
            input_money.append(ttt)
        else:
            sum_money_mmm.append(sum_money[i])
            input_money.append(ttt+input_month_total)
        pre_m = m   
    sum_money_mmm.append(sum_money[-1])
    print(sum_money_mmm)
    
    for i in range(len(input_money)):
        if input_money[i]==true_first_in:
            iiiii = i
        else:
            break
    # total_input_money = first_input_total+input_month_total*(len(sum_money_mmm)-1)
    total_input_money = true_first_in
    # print(sum_money[-1],total_input_money)
    final_sum = sum_money[iiiii]
    reward = (final_sum-total_input_money)/total_input_money
    print('reward',reward)
    # ans_reward.append(reward)
    annual_reward = math.pow( (1+reward), 1/nnnn )-1
    print('annual reward',annual_reward)
    # ans_ann_reward.append(annual_reward)

    m_rews = []

    for i in range(len(sum_money_mmm)):
        if i==0:
            input = true_first_in
        else:
            input = sum_money_mmm[i-1]
        # print(sum_money[i],input)
        rrr = (sum_money_mmm[i]-input)/input
        m_rews.append(rrr)

    # print(m_rews)
    std_dev_m = statistics.stdev(m_rews)
    v1_std = std_dev_m * math.pow( 12, 0.5 )
    print('std_dev',v1_std)
    # ans_dev.append(v1_std)

    mdd = cal_mdd(choose,weight,df2,ratios)
    print('mdd',mdd)
    # mdd = 0 

    return [reward,annual_reward,v1_std,mdd,sum_money[:iiiii+1],input_money[:iiiii+1]] # ,success_ratio,success

# %%
def test_choose_nochange(y,month,nnn_month,choose,weight,first_input_total,input_month_total,mode=3):
    print(choose,weight)
    nnnn = nnn_month/12
    if y+nnnn>today.year:
        print('y+nnnn too big')
        return [-100,-100,100,100,[],[]] #[reward,annual_reward,v1_std,mdd,sum_money]

    start_date = datetime.date(y,month,1)
    final_date = start_date + relativedelta(months=nnn_month)
    
    while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
        start_date += datetime.timedelta(days=1)
    # print(start_date)
    start_date_index = list(close_df['date']).index(start_date)
    while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
        final_date -= datetime.timedelta(days=1)
    print(start_date,final_date)
    final_date_index = list(close_df['date']).index(final_date)


    df = close_df[start_date_index:final_date_index+1]
    df = df.reset_index(drop=True)
    tmp=['date']
    for i in range(len(choose)):
        tmp.append(choose[i])

    df = df[tmp]
    # print(df)
    
    for i in range(len(df)):
        have_nan = True
        for a in range(len(choose)):
            if np.isnan(df[choose[a]][i])==True: #是空的
                have_nan = False
                break
        if have_nan==False:
            # print(df['date'][i])
            df = df.drop([i],axis=0)          
    df2 = df.reset_index(drop=True)

    if len(df2)==0:
        # print(df2)
        print('not create')
        return [-100,-100,100,100,[],[]]

    sum_money,ratios = money_sim(choose,weight,df2,start_date,mode,first_input_total,input_month_total)
    # print(sum_money)
    # print(len(ratios))
    # print(len(df2))

    input_money = [first_input_total]
    sum_money_mmm=[sum_money[0]]
    pre_m = df2['date'][0].month
    for i in range(1,len(sum_money)):
        m = df2['date'][i].month
        ttt = input_money[-1]
        if m==pre_m:
            input_money.append(ttt)
        else:
            sum_money_mmm.append(sum_money[i])
            input_money.append(ttt+input_month_total)
        pre_m = m   
    sum_money_mmm.append(sum_money[-1])
    print(sum_money_mmm)
    
    total_input_money = first_input_total + input_month_total*(nnn_month-1)
    for i in range(len(input_money)):
        if input_money[i]==total_input_money:
            iiiii = i
        elif input_money[i]>total_input_money:
            break
    # total_input_money = first_input_total+input_month_total*(len(sum_money_mmm)-1)
    
    # total_input_money = input_money[-1]
    # print(sum_money[-1],total_input_money)
    final_sum = sum_money[iiiii]
    reward = (final_sum-total_input_money)/total_input_money
    print('reward',reward)
    # ans_reward.append(reward)
    annual_reward = math.pow( (1+reward), 1/nnnn )-1
    print('annual reward',annual_reward)
    # ans_ann_reward.append(annual_reward)

    m_rews = []

    for i in range(len(sum_money_mmm)):
        if i==0:
            input = first_input_total
        else:
            input = sum_money_mmm[i-1]
        # print(sum_money[i],input)
        rrr = (sum_money_mmm[i]-input)/input
        m_rews.append(rrr)

    # print(m_rews)
    std_dev_m = statistics.stdev(m_rews)
    v1_std = std_dev_m * math.pow( 12, 0.5 )
    print('std_dev',v1_std)
    # ans_dev.append(v1_std)

    mdd = cal_mdd(choose,weight,df2,ratios)
    print('mdd',mdd)
    # mdd = 0 

    return [reward,annual_reward,v1_std,mdd,sum_money[:iiiii+1],input_money[:iiiii+1]] # ,success_ratio,success


# %%
def cal_mdd(choose,weight,df2,ratios):
    # print(df2)
    # mdds = []
    # 平均股價
    df2['avg'] = 0
    # df2 = df2.drop(['avg'],axis=1)
    for i in range(len(df2['avg'])):
        # print(ratios[i])
        for a in range(len(choose)):
            # df2.loc[i,'avg'] += df2[choose[a]][i]*weight[a]
            df2.loc[i,'avg'] += df2[choose[a]][i]*ratios[i][a]
    # print(df2)
    
    # 漲幅
    df2['day_return'] = 0 
    for i in range(len(df2)-1):
        df2.loc[i+1,'day_return'] = (df2['avg'][i+1] - df2['avg'][i])/df2['avg'][i]
    # print(df2)
    df2 = df2.fillna(0)
    # 無風險利率
    # risk_free_return = 0.01/365
    # risk_free_return = 0
    
    # avg_return = statistics.mean(df2['day_return'])

    df2['max']=0
    s1 = df2['avg']
    for i in range(len(df2)):
        df2.loc[i,'max'] = s1[0:i+1].max() 
    
    df2['dd'] = 0
    df2['dd'] = 1-(df2['avg']/df2['max'])
    
    mdd = df2['dd'].max()
    # print('mdd',mdd)
    # mdds.append(mdd)

    return mdd

# %%

def money_sim(choose,weight,df2,start_date,mode,first_input_total,input_month_total):
    # choose = []
    # for a in range(len(choose_obj)):
    #     choose.append(choose_obj[a])

    # print(df2)
    # moneys=[]
    #再平衡區
    balence_zone = 0.2
    #可容忍區
    tolerance_zone = 0.1
    #調整方式
    # mode = 3

    #手續費
    process_fee_percent = 0.1425/100

    start_month = start_date.month
    start_day = start_date.day
    transcation_day = start_day
    div_month = start_month

    manage_fee_days=[]
    div_percents=[]
    for a in range(len(choose)):
        sql = "select * from detail where name = '"+choose[a]+"'"
        # print(sql)
        cursor.execute(sql)
        result_select = cursor.fetchall()
        db.commit()
        # print(result_select)
        
        
        #總費用率費(內扣)
        manage_fee = float(result_select[0][13])#從性質表拿總費用率
        manage_fee_day = manage_fee /100/252#每天平均的內扣
        # print(manage_fee)
        manage_fee_days.append(manage_fee_day)
        
        #配息率
        div_percent = float(result_select[0][5])#從性質表拿配息率
        # print(div_percent)
        div_percents.append(div_percent)

    ratios = []
    moneys =    []
    units =     []
    sum_moneys =    []

    for i in range(len(df2)):
        money =    []
        unit =     []
        for a in range(len(choose)):
            
            #總費用率費(內扣)
            manage_fee_day = manage_fee_days[a]
            
            #配息率
            div_percent = div_percents[a]

            input_month = input_month_total*weight[a]
            
            
            
            if i==0:
                
                first_input = first_input_total*weight[a]
                start_unit = first_input/df2[choose[a]][0]
                unit.append(start_unit)
                money.append(first_input)
                # sum_money = sum(money) 
                # continue
            else:
                now_month = df2['date'][i].month
                now_day = df2['date'][i].day
                pre_day = df2['date'][i-1].day
                now_money = moneys[-1][a]
                now_unit = units[-1][a]
                now_close = df2[choose[a]][i]

                #每月投入錢
                if (pre_day<transcation_day and now_day>=transcation_day) or (pre_day>now_day and now_day>=transcation_day):
                    if now_month == div_month:#投入配息
                        buy_unit = (input_month + now_money*div_percent- manage_fee_day - process_fee_percent*input_month )/now_close
                        # print('month + div input',buy_unit)
                    else:
                        buy_unit = (input_month - manage_fee_day - process_fee_percent*input_month )/now_close

                    now_unit += buy_unit    
                    now_money = now_unit * now_close 
                    money.append(now_money) 
                    unit.append(now_unit)
                else:
                    # now_unit = unit[i] #今天的單位數=昨天的單位數
                    now_money = now_unit * now_close - manage_fee_day
                    now_unit = now_money/now_close
                    money.append(now_money) #將目前持有資金金額存入陣列以便觀察
                    unit.append(now_unit)    

        sum_money = sum(money) 
        # print(df2['date'][i],sum_money)
        # print(money)


        # 調整前list
        before_adj_ratio = []
        for z in range(len(choose)):
            before_adj_ratio.append(money[z]/sum_money)
        # print(total_cost)
    
        balence_array = []
        for z in range(len(weight)):
            balence_array.append(weight[z]*balence_zone)
        up_range = []
        for z in range(len(weight)):
            up_range.append(weight[z]+balence_array[z])
        down_range = []
        for z in range(len(weight)):
            down_range.append(weight[z]-balence_array[z])
        tolerance_array = []
        for z in range(len(weight)):
            tolerance_array.append(weight[z]*tolerance_zone)
        up_range_tolerance = []
        for z in range(len(weight)):
            up_range_tolerance.append(weight[z]+tolerance_array[z])
        down_range_tolerance = []
        for z in range(len(weight)):
            down_range_tolerance.append(weight[z]-tolerance_array[z]) 
    
        sell_buy = []
        # 調整後list
        after_adj_ratio = []
        # mode = 4
        if mode == 1 :
            for z in range(len(before_adj_ratio)):
                if (before_adj_ratio[z] > up_range[z] or before_adj_ratio[z] < down_range[z]):
                    # print(df2['date'][i],'rebalnce')
                    sell_buy.append(weight[z]-before_adj_ratio[z])
                    after_adj_ratio.append(weight[z]) 
                else:
                    sell_buy.append(0)
                    after_adj_ratio.append(before_adj_ratio[z]) 
        elif mode == 2:
            for z in range(len(before_adj_ratio)):
                if (before_adj_ratio[z] > up_range[z]):
                    # print(df2['date'][i],'rebalnce')
                    sell_buy.append(up_range[z]-before_adj_ratio[z])
                    after_adj_ratio.append(up_range[z]) 
                elif (before_adj_ratio[z] < down_range[z]):
                    # print(df2['date'][i],'rebalnce')
                    sell_buy.append(down_range[z]-before_adj_ratio[z])
                    after_adj_ratio.append(down_range[z]) 
                else:
                    sell_buy.append(0)
                    after_adj_ratio.append(before_adj_ratio[z])
        elif mode == 3:
            for z in range(len(before_adj_ratio)):
                if (before_adj_ratio[z] > up_range[z]):
                    # print(df2['date'][i],'rebalnce')
                    sell_buy.append(up_range_tolerance[z]-before_adj_ratio[z])
                    after_adj_ratio.append(up_range_tolerance[z]) 
                elif (before_adj_ratio[z] < down_range[z]):
                    # print(df2['date'][i],'rebalnce')
                    sell_buy.append(down_range_tolerance[z]-before_adj_ratio[z])
                    after_adj_ratio.append(down_range_tolerance[z]) 
                else:
                    sell_buy.append(0)
                    after_adj_ratio.append(before_adj_ratio[z]) 
        elif mode==4:
            for z in range(len(before_adj_ratio)):
                sell_buy.append(0)
                after_adj_ratio.append(before_adj_ratio[z]) 
        # print(df2['date'][i],after_adj_ratio)
        new_asset = []
        for z in range(len(choose)):
            tmp = after_adj_ratio[z]*sum_money
            new_asset.append(tmp)   
            money[z] = tmp
        ratios.append(after_adj_ratio)
        moneys.append(money)
        units.append(unit)
        sum_moneys.append(sum_money)
    # print(moneys)
    # print(units)
    # print(sum_moneys)    
    return sum_moneys,ratios

# %%

# def cal_success_ratio(y,nnnn,choose,weight,expect_r,mode,first_input_total):
#     # sql_today = close_df['date'][len(close_df)-1]
#     # tmp = [1,4,7,10]
#     success = []
#     stat_list = []
#     start_date = datetime.date(y,1,1)
#     final_date = datetime.date(y,4,1)
#     final_day = datetime.date(y+nnnn,1,1)

#     while (final_day-final_date).days >=0:
#         final_date_tmp = final_date

#         while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
#             start_date += datetime.timedelta(days=1)
#         # print(start_date)
#         start_date_index = list(close_df['date']).index(start_date)
#         while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
#             final_date -= datetime.timedelta(days=1)
#         # print(start_date,final_date)
#         final_date_index = list(close_df['date']).index(final_date)

#         if (final_day-final_date).days<0:
#             break

#         df = close_df[start_date_index:final_date_index+1]
#         df = df.reset_index(drop=True)
#         tmp=['date']
#         for i in range(len(choose)):
#             tmp.append(choose[i])
    
#         df = df[tmp]
        
#         for i in range(len(df)):
#             have_nan = True
#             for a in range(len(choose)):
#                 if np.isnan(df[choose[a]][i])==True: #是空的
#                     have_nan = False
#                     break
#             if have_nan==False:
#                 # print(df['date'][i])
#                 df = df.drop([i],axis=0)          
#         df2 = df.reset_index(drop=True)

#         sum_money,ratios = money_sim(choose,weight,df2,start_date,mode)

#         if len(df2)<=0:
#             continue
#         len_m=3
#         # pre_m = df2['date'][0].month
#         # for i in range(1,len(sum_money)):
#         #     m = df2['date'][i].month
#         #     if m==pre_m:
#         #         pass
#         #     else:
#         #         len_m+=1
#         #     pre_m = m

#         total_input_money = first_input_total+input_month_total*( len_m)
#         # print(len_m,len(sum_money)  ,sum_money)
#         print(sum_money[-1],total_input_money)
#         reward = (sum_money[-1]-total_input_money)/total_input_money
#         # print('reward',reward)
#         annual_reward = math.pow( (1+reward), 1/0.25 )-1
#         print(start_date,final_date,annual_reward)

        
#         start_date = final_date_tmp    
#         final_date = start_date + relativedelta(months=3)

#         if annual_reward >=expect_r:
#             success.append(1)
#         else:
#             success.append(0)
        
#         tmp = (annual_reward-expect_r)/expect_r
#         stat_list.append(tmp)

#     print('success',success)
#     success_ratio = sum(success)/len(success)
#     print('success_ratio',success_ratio)
#     stat = sum(stat_list)/len(stat_list)
#     print('stat_list',stat_list)
#     print('stat',stat)

#     return success_ratio,success,stat,stat_list



# # %%
# def slidwindow(y,nnnn,choose,weight,expect_r,mode,first_input_total):
#     print('slidwindow:')
#     # train y-n/1/1 ~ y-1/12/31
#     success = []
#     stat_list = []
#     for yyyy in range(2010,2020-nnnn+2):
#         # yyyy=2010
#         # if yyyy == (y-nnnn):
#         #     continue
#         start_date = datetime.date(yyyy,1,1)
#         final_date = datetime.date(yyyy+nnnn-1,12,31)

#         while (start_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(start_date),"%Y-%m-%d")<datetime.datetime.strptime(str(today),"%Y-%m-%d") ):
#             start_date += datetime.timedelta(days=1)
#         # print(start_date)
#         start_date_index = list(close_df['date']).index(start_date)
#         while (final_date in list(close_df['date']))==False and ( datetime.datetime.strptime(str(final_date),"%Y-%m-%d")>datetime.datetime.strptime(str(start_date),"%Y-%m-%d") ):
#             final_date -= datetime.timedelta(days=1)
#         print(start_date,final_date)
#         final_date_index = list(close_df['date']).index(final_date)
    
    
#         df = close_df[start_date_index:final_date_index+1]
#         df = df.reset_index(drop=True)
#         tmp=['date']
#         for i in range(len(choose)):
#             tmp.append(choose[i])
    
#         df = df[tmp]
        
#         for i in range(len(df)):
#             have_nan = True
#             for a in range(len(choose)):
#                 if np.isnan(df[choose[a]][i])==True: #是空的
#                     have_nan = False
#                     break
#             if have_nan==False:
#                 # print(df['date'][i])
#                 df = df.drop([i],axis=0)          
#         df2 = df.reset_index(drop=True)

#         sum_money,ratios = money_sim(choose,weight,df2,start_date,mode)

#         # sum_money_mmm=[sum_money[0]]
#         # print(len(df2))
#         if len(df2)<=0:
#             continue
#         len_m=1
#         pre_m = df2['date'][0].month
#         for i in range(1,len(sum_money)):
#             m = df2['date'][i].month
#             if m==pre_m:
#                 # df = df.drop([i],axis=0)
#                 pass
#             else:
#                 # sum_money_mmm.append(sum_money[i])
#                 len_m+=1
#             pre_m = m   
#         # sum_money_mmm.append(sum_money[-1])
#         # print(sum_money_mmm)

#         total_input_money = first_input_total+input_month_total*( len_m)
#         # print(len_m )
#         # print(sum_money[-1],total_input_money)
#         reward = (sum_money[-1]-total_input_money)/total_input_money
#         # print('reward',reward)
#         # ans_reward.append(reward)
#         annual_reward = math.pow( (1+reward), 1/nnnn )-1
#         # print('annual reward',annual_reward)
        
#         if annual_reward >=expect_r:
#             success.append(1)
#         else:
#             success.append(0)
#         tmp = (annual_reward-expect_r)/expect_r
#         stat_list.append(tmp)    
#     print('success',success)
#     success_ratio = sum(success)/len(success)
#     print('success_ratio',success_ratio)
#     stat = sum(stat_list)/len(stat_list)
#     print('stat_list',stat_list)
#     print('stat',stat)

#     return success_ratio,success,stat,stat_list        
# %%
def get_all_value(y,month,choose,weight,first_input_total,true_first_in,mode=4):
    ans = test_choose(y,month,choose,weight,first_input_total,true_first_in,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
    # success_ratio_all,success_all = slidwindow(y,nnnn,choose,weight,ans[1],mode)
    if ans is None:
        return ans 
    # success_ratio,success,stat = cal_success_ratio(y,nnnn,choose,weight,ans[1],mode)
    ## print(ans,success_ratio,success)
    # tmp = []
    ## ans.append(success_ratio_all)
    # ans.append(success_ratio)

    return ans 



# %%
# choose1 = 'VHT DVY XLV VBK IWF'
# weight1 = '0.20000 0.20000 0.20000 0.20000 0.20000'

# choose = choose1.split(' ')
# weight = weight1.split(' ')
# for i in range(len(weight)):
#     weight[i] = float(weight[i])
# %%
# train y-nnnn/1/1 ~ y-1/12/31
# test y/1/1 ~ y+nnnn-1/12/31

# y=2017 #test start year
# nnnn=3 
# expect_r = 0.08
# mode = 3 #rebalance mode

# ans = test_choose(y,nnnn,choose,weight,expect_r,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
# success_ratio,success = slidwindow(y,nnnn,choose,weight,expect_r,mode)
# success_ratio,success = cal_success_ratio(y,nnnn,choose,weight,expect_r,mode)
# print(ans,success_ratio,success)
# %%
# ans_list = ['reward','annual_reward','v1_std','mdd','success_ratio','success']
# print()
# print('answers:')
# for i in range(1,len(ans)-1):
#     print(ans_list[i],ans[i])

# %%



# %%
# ans = test_choose(y,nnnn,choose,weight,expect_r,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
# success_ratio,success = slidwindow(y,nnnn,choose,weight,expect_r,mode)
# success_ratio,success = cal_success_ratio(y,nnnn,choose,weight,expect_r,mode)
# print(ans,success_ratio,success)