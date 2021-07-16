# %%
# import clustering_corr
# import clustering_type
import datetime
import time
import numpy as np
import random
import generate_input_data
import rnn
import bl_weight
import price2matrix

# %%
if __name__ == '__main__':
    start = time.time()

    db_name = 'my_etf'
    list_etf = ['TW_etf']
    (y,expect_reward,nnnn,month) = (2018,0.08,1,6) # 2018/1/1~2018/12/31
    market_etf = '0050.TW'
    number = 3
    # cluster = 'type'
    cluster = 'corr'

    print('generate data')
    closes,volumes,volatilitys,groups,number = generate_input_data.generate_data(y,expect_reward,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    train_closes,train_volumes,train_volatilitys = generate_input_data.generate_training_data(y,month,db_name,groups,number)

    
    batch_size = 10
    hidden_layer = 256
    clip_margin = 5
    learning_rate = 0.001
    epochs = 100
    window_size = 21

    mse_record = []
    predict_record = []

    print('開始訓練與預測')
    # for i in range(number):
    #     if i == 0:
    #         print('大盤')
    #     else:
    #         print('第i群')
        
    #     data_to_use = np.array( [ train_closes[i],train_volumes[i],train_volatilitys[i] ] )
    #     data_a_month = np.array([ closes[i], volumes[i], volatilitys[i] ] )
        
    #     mse, predict_price = rnn.train_lstm(batch_size,hidden_layer,clip_margin,learning_rate,epochs,window_size,data_to_use,data_a_month)
    #     mse_record.append(mse)
    #     predict_record.append(predict_price)
    
    # predict_record = np.array(predict_record)
    # mse_record = np.array(mse_record)

    predict_record = np.array([83,30,13,12])
    mse_record = np.array([0.1,0.15,0.13,0.12])
    print('predict_record',predict_record)
    print('mse_record',mse_record)

    P, Q, omega, cov_matrix = price2matrix.generate_matrix(closes,number,predict_record,mse_record)
    
    # print('P',P)
    # print('Q',Q)
    # print('omega',omega)
    # print('cov_matrix',cov_matrix)

    date1 = datetime.date(y,month,1)
    date2 = datetime.date(y,month+1,1)
    w = bl_weight.get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    weights = []
    for i in range(len(w)):
        weights.append(w[i])
    weights = np.array(weights)
    print('weights',weights)


    # 抽籤
    max_n = round(np.min(weights/0.05))
    ran_n = []
    for j in range(1,number):
        tmp = []
        for k in range(len(groups[j])):
            tmp.append(k)
        random.shuffle(tmp)
        ran_n.append(tmp)

    combs = []
    for i in range(1,int(max_n)+1):
        c = []
        for j in range(number-1):
            c.append([])
        for j in range(number-1):
            for k in range(i):
                c[j].append(groups[j+1][ran_n[j][k]])
        # print(c)
        combs.append(c)
    # print(combs)

    print('answers:')
    ans = []
    for i in range(len(combs)):
        w = ''
        names = ''
        for j in range(number-1):
            for k in range(len(combs[i][j])):
                names += (combs[i][j][k]+' ')
                tmp_w = round(weights[j]/len(combs[i][j]) , 5)
                w += (str(tmp_w)+' ')
        ans.append([names[:-1],w[:-1]])   
    for i in range(len(ans)):
        print(ans[i]) 

    

    end = time.time()
    print(end-start)

# %%
