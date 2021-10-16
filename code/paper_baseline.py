# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import generate_input_data
import time
import math
import sys
# %%

def calculate(closes):

    price_data = pd.DataFrame(closes).T

    log_ret = np.log(price_data/price_data.shift(1))

    cov_mat = log_ret.cov() * 252
    print(cov_mat)

    # Simulating 5000 portfolios
    num_port = 5000
    # Creating an empty array to store portfolio weights
    all_wts = np.zeros((num_port, len(price_data.columns)))
    # Creating an empty array to store portfolio returns
    port_returns = np.zeros((num_port))
    # Creating an empty array to store portfolio risks
    port_risk = np.zeros((num_port))
    # Creating an empty array to store portfolio sharpe ratio
    sharpe_ratio = np.zeros((num_port))


    for i in range(num_port):
        
        flag=0
        while flag == 0:
            wts = np.random.uniform(size = len(price_data.columns))
            wts = wts/np.sum(wts)
            if (np.max(wts)<0.5) and (np.min(wts)>0.05):
                flag = 1

        # saving weights in the array

        all_wts[i,:] = wts

        # Portfolio Returns

        port_ret = np.sum(log_ret.mean() * wts)
        port_ret = (port_ret + 1) ** 252 - 1

        # Saving Portfolio returns

        port_returns[i] = port_ret


        # Portfolio Risk

        port_sd = np.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))

        port_risk[i] = port_sd

        # Portfolio Sharpe Ratio
        # Assuming 0% Risk Free Rate

        sr = port_ret / port_sd
        sharpe_ratio[i] = sr
    
    return all_wts,port_risk,sharpe_ratio,port_returns

# 找到最低標準差
def mvp(closes):

    all_wts,port_risk,sharpe_ratio,port_returns = calculate(closes)
    min_var = all_wts[port_risk.argmin()]
    var = port_risk[port_risk.argmin()]
    reward = port_returns[port_risk.argmin()]
    # print(min_var)
    return min_var,var,reward

# 找到最大夏普值
def mvtp(closes):

    all_wts,port_risk,sharpe_ratio,port_returns = calculate(closes)
    max_sr = all_wts[sharpe_ratio.argmax()]
    var = port_risk[sharpe_ratio.argmax()]
    reward = port_returns[sharpe_ratio.argmax()]
    # print(max_sr)
    return max_sr,var,reward

# %%
# 選各群代表、產出組合(第一次選組合)
def draw_target(weights,number,groups,every_close_finalday):
    # 抽籤
    for i in range(len(weights)):
        if weights[i]<0:
            weights[i]=0
    sum_w = np.sum(weights)
    weights = weights/sum_w

    for i in range(len(weights)):
        if weights[i]<0.05 and weights[i]>0:
            weights[i]=0.05
    sum_w = np.sum(weights)
    weights = weights/sum_w

    tmp_w = weights.copy()
    for i in range(len(tmp_w)):
        if tmp_w[i]==0:
            tmp_w[i]=10000
    if np.min(tmp_w)<=0.05:
        max_n = 1
    else:
        max_n = math.ceil(np.min(tmp_w/0.05))
    print(max_n)
    # 群內排序
    # ran_n = []
    # for j in range(1,number):
    #     tmp = []
    #     for k in range(len(groups[j])):
    #         tmp.append(k)
    #     random.shuffle(tmp)
    #     ran_n.append(tmp)

    ran_n = []
    # ran_n.append([])
    for j in range(1,number):
        # tmp = []
        sort_id = sorted(range(len(every_close_finalday[j])), key=lambda k: every_close_finalday[j][k], reverse=True)
        ran_n.append(sort_id)

    combs = []
    for i in range(1,int(max_n)+1):
        c = []
        for j in range(number-1):
            c.append([])
        for j in range(number-1):
            # if tmp_w[j]==10000:#此類0%
                # c[j].append(groups[j+1][ran_n[j][0]])
                # continue
            for k in range(i):
                if len(ran_n[j])>k:
                    c[j].append(groups[j+1][ran_n[j][k]])
        # print(c)
        combs.append(c)
    # print(combs)

    # print('answers:')
    ans = []
    for i in range(len(combs)):
        w = ''
        names = ''
        for j in range(number-1):
            for k in range(len(combs[i][j])):
                names += (combs[i][j][k]+' ')
                if weights[j]==10000:
                    tmp_w = 0.0
                else:
                    tmp_w = round(weights[j]/len(combs[i][j]) , 5)
                w += (str(tmp_w)+' ')
        ans.append([names[:-1],w[:-1]])   
    # for i in range(len(ans)):
    #     print(ans[i]) 
    
    return ans

# %%
# 產出組合(動態調整權重)
def draw_target_2(weights,number,groups,every_close_finalday):
    # 抽籤
    for i in range(len(weights)):
        if weights[i]<0:
            weights[i]=0
    sum_w = np.sum(weights)
    weights = weights/sum_w

    for i in range(len(weights)):
        if weights[i]<0.05 and weights[i]>0:
            weights[i]=0.05
    sum_w = np.sum(weights)
    weights = weights/sum_w

    str_w = ''
    # names = ' '.join(groups[0])
    names = ''
    for i in range(len(weights)):
        str_w += (str(round(weights[i], 5))+' ')
        names += (groups[i+1][0]+' ')

    ans = [[names[:-1],str_w[:-1]]]
    
    return ans

# %%

# 產生第一個組合(分群、預測、分配權重、選代表)
def choose_target(db_name,list_etf,y,nnnn,month,market_etf,number,cluster,predict_type):
    
    print('generate data')
    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    train_closes,train_volumes,train_volatilitys = generate_input_data.generate_training_data(y,month,db_name,groups,number)
    
    if predict_type == 'mvp':
        weights,var,reward = mvp(closes[1:])
    elif predict_type == 'mvtp':
        weights,var,reward = mvtp(closes[1:])

    # weights = []
    # for i in range(len(w)):
    #     weights.append(w[i])
    weights = np.array(weights)
    print('weights',weights)

    # print('answers')
    ans = draw_target(weights,number,groups,every_close_finalday)
    # for i in range(len(ans)):
    #     print(ans[i]) 

    return ans

# %%
# 動態調整組合(根據輸入的組合調權重)
def dynamic_target(groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,predict_type):

    print('generate data')
    # closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data_d(y,nnnn,month,db_name,cluster,number,market_etf,list_etf,groups)
    # train_closes,train_volumes,train_volatilitys = generate_input_data.generate_training_data(y,month,db_name,groups,number)

    if predict_type == 'mvp':
        weights,var,reward = mvp(closes[1:])
    elif predict_type == 'mvtp':
        weights,var,reward = mvtp(closes[1:])

    weights = np.array(weights)
    print('weights',weights)

    # print(groups)

    # print('answers')
    ans = draw_target_2(weights,number,groups,every_close_finalday)
    # for i in range(len(ans)):
    #     print(ans[i]) 
    return ans

# %%
if __name__ == '__main__':
    start = time.time()

    db_name = 'my_etf'
    # list_etf = ['TW_etf']
    # market_etf = '006204.TW'
    # market = 'tw'
    
    list_etf = ['US_etf']
    market_etf = 'SPY'
    market = 'us'

    (y,nnnn,month) = (2018,1,6) # 2018/1/1~2018/12/31
    
    cluster = 'corr'

    filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/mvp-3y/'
    predict_type = 'mvp'
    # filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/mvtp-3y/'
    # predict_type = 'mvtp'

    # ans_list = choose_target(db_name,list_etf,y,expect_reward,nnnn,month,market_etf,number,cluster,predict_type)
    # print(ans_list)
    
    # first_y = 2015
    # first_month = 1
    first_y = int(sys.argv[1])
    first_month = int(sys.argv[2])
    part = int(sys.argv[3])
    run_len = int(sys.argv[4])
    print(first_y,first_month,part,run_len)
############################### 一般演算法 ##################################

    if part==1:
        count = 0
        while count<4:

            # 設定初始年/月、分群數量，產出第1個組合並動態平衡11次，最後一次會存檔

            number = 5
            count += 1

            print(str(first_y)+'/'+str(first_month))

            y = first_y
            month = first_month

            ans_list = choose_target(db_name,list_etf,first_y,nnnn,first_month,market_etf,number,cluster,predict_type)
            # ans_list = [['VTI VGK VPL VWO BWX IEI','0.15 0.15 0.15 0.15 0.2 0.2']]
            ans = [ans_list[0]]
            print(ans)


            df_list = [[y,month,ans[0][0],ans[0][1]]]
            # df_list = []

            groups = [[market_etf]]
            tmp_g = ans_list[0][0].split(' ')
            for i in range(len(tmp_g)):
                groups.append([tmp_g[i]])
            number = len(groups)

            # ans_new_list = []
            for j in range(run_len-1):
                if month!=12:
                    month += 1
                else:
                    y+=1
                    month=1

                ans_new = dynamic_target(groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,predict_type)
                # ans_new_list.append(ans_new[0])
                tmp = [y,month,ans_new[0][0],ans_new[0][1]]
                print(tmp)
                df_list.append(tmp)

                df = pd.DataFrame(df_list,columns=['year','month','names','weights'])
                df.to_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv',index=False)

            # print(ans)
            # print(ans_new_list)
            print(df_list)

            df = pd.DataFrame(df_list,columns=['year','month','names','weights'])
            df.to_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv',index=False)

            if first_month+12>=12:
                first_y+=1
                first_month=(first_month+12)-12
            else:
                first_month+=12

############################## 固定標的 只調權重 ####################################
    if part==3:

        count = 0
        while count<1:
            
            # 設定初始年/月、分群數量，產出第1個組合並動態平衡11次，最後一次會存檔
    
            number = 5
            count += 1
    
            print(str(first_y)+'/'+str(first_month))
    
            y = first_y
            month = first_month
    
            ans_list = [['ITOT VEU VNQ AGG','0.36 0.18 0.06 0.4']]
            ans = [ans_list[0]]
            print(ans)
    
            df_list = []
    
            groups = [[market_etf]]
            tmp_g = ans_list[0][0].split(' ')
            for i in range(len(tmp_g)):
                groups.append([tmp_g[i]])
            number = len(groups)
    
            for j in range(run_len):
                
                ans_new = dynamic_target(groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,predict_type)
                tmp = [y,month,ans_new[0][0],ans_new[0][1]]
                print(tmp)
                df_list.append(tmp)

                df = pd.DataFrame(df_list,columns=['year','month','names','weights'])
                df.to_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv',index=False)
    
                if month!=12:
                    month += 1
                else:
                    y+=1
                    month=1
    
            print(df_list)
    
            df = pd.DataFrame(df_list,columns=['year','month','names','weights'])
            df.to_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv',index=False)

            if first_month+12>=12:
                first_y+=1
                first_month=(first_month+12)-12
            else:
                first_month+=12

    end = time.time()
    print(end-start)



# %%
