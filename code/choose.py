# %%
import os
import datetime
from os import name
import time
import numpy as np
import random
import math
from numpy.lib.function_base import append
import pandas as pd 
import generate_input_data
import new_rnn
import bl_weight
import price2matrix
import sys

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
def choose_target(lstm_filepath,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type):
    
    print('generate data')
    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    train_closes,train_volumes,train_volatilitys = generate_input_data.generate_training_data(y,month,db_name,groups,number)

    mse_record = []
    predict_record = []

    print('開始訓練與預測')
    for i in range(number):
        if i == 0:
            print('大盤')
            filename = str(y)+'-'+str(month)+' market.png'
        else:
            print('第'+str(i)+'群')
            filename = str(y)+'-'+str(month)+' cluster-'+str(i)+'.png'
        
        data_to_use = np.array( [ train_closes[i],train_volumes[i],train_volatilitys[i] ] )
        data_a_month = np.array([ closes[i], volumes[i], volatilitys[i] ] )
        
        # mse, predict_price = rnn.train_lstm(batch_size,hidden_layer,clip_margin,learning_rate,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        if lstm_type=='lstm':
            mse, predict_price = new_rnn.lstm(batch_size,hidden_layer,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        else:
            mse, predict_price = new_rnn.ecm_lstm(batch_size,hidden_layer,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        mse_record.append(mse)
        predict_record.append(predict_price)
    
    predict_record = np.array(predict_record)
    mse_record = np.array(mse_record)

    # predict_record = np.array([83,30,13,12])
    # mse_record = np.array([0.1,0.15,0.13,0.12])
    # print('predict_record',predict_record)
    print('predict_price',predict_record)
    print('mse_record',mse_record)

    P, Q, omega, cov_matrix = price2matrix.generate_matrix(closes,number,predict_record,mse_record)
    # print('P',P)
    # print('Q',Q)
    # print('omega',omega)
    # print('cov_matrix',cov_matrix)

    date1 = datetime.date(y,month,1)
    if month==12:
        date2 = datetime.date(y+1,1,1)
    else:
        date2 = datetime.date(y,month+1,1)
    
    w = bl_weight.get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    weights = []
    for i in range(len(w)):
        weights.append(w[i])
    weights = np.array(weights)
    print('weights',weights)

    # print('answers')
    ans = draw_target(weights,number,groups,every_close_finalday)
    # for i in range(len(ans)):
    #     print(ans[i]) 

    return ans,groups,number

# %%
# 動態調整組合(根據輸入的組合調權重)
def dynamic_target(lstm_filepath,groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type):
    
    print('generate data')
    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data_d(y,nnnn,month,db_name,cluster,number,market_etf,list_etf,groups)
    # closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data(y,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    train_closes,train_volumes,train_volatilitys = generate_input_data.generate_training_data(y,month,db_name,groups,number)

    mse_record = []
    predict_record = []

    print('開始訓練與預測')
    for i in range(number):
        if i == 0:
            print('大盤')
            filename = str(y)+'-'+str(month)+' market.png'
        else:
            print('第'+str(i)+'群')
            filename = str(y)+'-'+str(month)+' cluster-'+str(i)+'.png'
        
        data_to_use = np.array( [ train_closes[i],train_volumes[i],train_volatilitys[i] ] )
        data_a_month = np.array([ closes[i], volumes[i], volatilitys[i] ] )
        
        # mse, predict_price = rnn.train_lstm(batch_size,hidden_layer,clip_margin,learning_rate,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        if lstm_type=='lstm':
            mse, predict_price = new_rnn.lstm(batch_size,hidden_layer,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        else:
            mse, predict_price = new_rnn.ecm_lstm(batch_size,hidden_layer,epochs,window_size,data_to_use,data_a_month,filename,lstm_filepath)
        mse_record.append(mse)
        predict_record.append(predict_price)
    
    predict_record = np.array(predict_record)
    mse_record = np.array(mse_record)

    # predict_record = np.array([83,30,13,12])
    # mse_record = np.array([0.1,0.15,0.13,0.12])
    # print('predict_record',predict_record)
    print('predict_price',predict_record)
    print('mse_record',mse_record)

    P, Q, omega, cov_matrix = price2matrix.generate_matrix(closes,number,predict_record,mse_record)
    # print('P',P)
    # print('Q',Q)
    # print('omega',omega)
    # print('cov_matrix',cov_matrix)

    date1 = datetime.date(y,month,1)
    if month==12:
        date2 = datetime.date(y+1,1,1)
    else:
        date2 = datetime.date(y,month+1,1)
    
    w = bl_weight.get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    weights = []
    for i in range(len(w)):
        weights.append(w[i])
    weights = np.array(weights)
    print('weights',weights)

    # print('answers')
    ans = draw_target_2(weights,number,groups,every_close_finalday)
    # for i in range(len(ans)):
    #     print(ans[i]) 

    return ans,groups,number
    

# %%

# 記憶體會爆掉 所以1年的模擬要分2次跑
if __name__ == '__main__':
    start = time.time()
    (y,nnnn,month) = (2018,1,6) # 2018/1/1~2018/12/31

    db_name = 'my_etf'
    
    list_etf = ['US_etf']
    market_etf = 'SPY'
    market = 'us'
    # list_etf = ['TW_etf']
    # market_etf = '006204.TW'
    # market = 'tw'

    # cluster = 'type'
    cluster = 'corr'

    # lstm_type = 'lstm'
    # filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/lstm/' # 存組合答案
    # fig_filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/lstm/' # 存lstm預測績效圖

    lstm_type = 'ecm'
    filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/ecm/' # 存組合答案
    fig_filepath = 'D:/Alia/Documents/asset allocation/output/answer/fix comb/ecm/' # 存lstm預測績效圖

    batch_size = 30
    hidden_layer = 64
    epochs = 100
    window_size = 21

    # 設定初始年/月
    # first_y = 2017
    # first_month = 1
    first_y = int(sys.argv[1])
    first_month = int(sys.argv[2])
    part = int(sys.argv[3])
    run_len = int(sys.argv[4])
    print(first_y,first_month,part,run_len)
############################# 第1部分 #########################################

    if part==1:

        # 設定分群數量，產出第1個組合並動態平衡3次，每次動態平衡後都會存檔

        count = 0
        while count<1:
            number = 5
            count += 1
            

            print(str(first_y)+'/'+str(first_month))

            y = first_y
            month = first_month

            lstm_filepath = fig_filepath+str(first_y)+'-'+str(first_month)+'/'
            os.mkdir(lstm_filepath)
            ans_list,groups,number = choose_target(lstm_filepath,db_name,list_etf,first_y,nnnn,first_month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type)
            ans = [ans_list[0]]
            print(ans)


            df_list = [[y,month,ans[0][0],ans[0][1]]]

            # groups = [[market_etf]]
            # tmp_g = ans_list[0][0].split(' ')
            # for i in range(len(tmp_g)):
            #     groups.append([tmp_g[i]])
            # number = len(groups)

            # ans_new_list = []
            for j in range(run_len-1):
                if month!=12:
                    month += 1
                else:
                    y+=1
                    month=1
                try:
                    ans_new,groups,number = dynamic_target(lstm_filepath,groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type)
                    tmp = [y,month,ans_new[0][0],ans_new[0][1]]
                except:
                    tmp = [y,month,tmp[2],tmp[3]]
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

# %%

############################# 第2部分 #########################################

    if part==2:
    # 讀入第1部分跑的結果，繼續跑動態平衡，每次跑完都會存檔

        count = 0
        while count<1:
            number = 5
            count += 1

            print(str(first_y)+'/'+str(first_month))
            y = first_y
            month = first_month
            lstm_filepath = fig_filepath+str(first_y)+'-'+str(first_month)+'/'
            # os.mkdir(lstm_filepath)

            ################################## 在這裡讀入第一部分輸出的結果 ######################################

            df = pd.read_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv')

            # ans_list = [['GUSH DPST LABD SOXS JDST', '0.0 0.24673 0.48312 0.09517 0.17498']] #第一個組合
            ans_list = [[df['names'][0],df['weights'][0]]]
            ans = [ans_list[0]]
            print(ans)

            # 第1部分的所有結果
            df_list = []
            for i in range(len(df)):
                tmp = [ int(df['year'][i]), int(df['month'][i]), df['names'][i], df['weights'][i] ] 
                df_list.append(tmp)
            print(df_list)

            # 第1部分跑的最後一個年/月
            y= int(df['year'][len(df)-1])
            month = int(df['month'][len(df)-1])

            ####################################################################################################

            groups = [[market_etf]]
            tmp_g = ans_list[0][0].split(' ')
            for i in range(len(tmp_g)):
                groups.append([tmp_g[i]])
            number = len(groups)
            # ans_new_list = []

            for j in range(run_len):
                if month!=12:
                    month += 1
                else:
                    y+=1
                    month=1
                print(y,month)
                try:
                    ans_new = dynamic_target(lstm_filepath,groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type)
                    tmp = [y,month,ans_new[0][0],ans_new[0][1]]
                except:
                    tmp = [y,month,tmp[2],tmp[3]]
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

# %%
############################## 第3部分 固定標的 #######################################

    if part==3:
    # 讀入固定的初始組合，繼續跑動態平衡，每次跑完都會存檔

        count = 0
        while count<1:
            number = 5
            count += 1

            print(str(first_y)+'/'+str(first_month))
            y = first_y
            month = first_month
            lstm_filepath = fig_filepath+str(first_y)+'-'+str(first_month)+'/'
            # os.mkdir(lstm_filepath)

            ans_list = [['ITOT VEU VNQ AGG','0.36 0.18 0.06 0.4']]
            # ans_list = [[df['names'][0],df['weights'][0]]]
            ans = [ans_list[0]]
            print(ans)

            df_list = []
            
            groups = [[market_etf]]
            tmp_g = ans_list[0][0].split(' ')
            for i in range(len(tmp_g)):
                groups.append([tmp_g[i]])
            number = len(groups)
            # ans_new_list = []

            for j in range(run_len):
                
                print(y,month)
                try:
                    ans_new = dynamic_target(lstm_filepath,groups,db_name,list_etf,y,nnnn,month,market_etf,number,cluster,batch_size,hidden_layer,epochs,window_size,lstm_type)
                    tmp = [y,month,ans_new[0][0],ans_new[0][1]]
                except:
                    tmp = [y,month,tmp[2],tmp[3]]
                print(tmp)
                df_list.append(tmp)
                df = pd.DataFrame(df_list,columns=['year','month','names','weights'])
                df.to_csv(filepath+'ans-'+market+'-'+str(first_y)+'-'+str(first_month)+'.csv',index=False)
                if month!=12:
                    month += 1
                else:
                    y+=1
                    month=1
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

    end = time.time()
    print(end-start)

# %%
