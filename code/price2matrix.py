# %%
from matplotlib.pyplot import close
import numpy as np
import datetime
import generate_input_data
import bl_weight

# %%
def generate_matrix_new(closes,number,price_pred,probibility):
    closes = np.array(closes)
    cov_matrix = np.corrcoef(closes[1:])
    price_now = []
    for i in range(1,number):
        if closes[i][0]!=0:
            price_now.append(closes[i][0])
        else:
            flag = 0
            for j in range(1,len(closes[i])):
                if closes[i][j]!=0:
                    price_now.append(closes[i][j])
                    flag = 1
                    break
            if flag==0:
                price_now.append(0.0001)
    price_now = np.array(price_now)
    
    price_change = []
    for i in range(number-1):
        tmp = []
        for j in range(3):
            ch = (price_pred[i][j]-price_now[i])/price_now[i]
            tmp.append(ch)
        price_change.append(tmp)
    price_change = np.array(price_change)
    confid = np.array(probibility).reshape((number-1)*3)
    # print(price_change.shape,confid.shape)
    print(price_change)
    print(confid)
    
    Q = []
    P = []
    omega = []
    for i in range(number-1):
        for k in range(3):
            Q.append(price_change[i][k])
            tmp = []
            for j in range(number-1):
                if j==i:
                    tmp.append(1)
                else:
                    tmp.append(0)
            P.append(tmp)

    
    for i in range((number-1)*3):
        tmp_o = []
        for j in range((number-1)*3):
            if i==j:
                if confid[i]!=0:
                    tmp_o.append(confid[i])
                else:
                    tmp_o.append(0.01)
            else:
                tmp_o.append(0)
        omega.append(tmp_o)    
        
    Q = np.array(Q).reshape(-1, 1)
    P = np.array(P)
    omega = np.array(omega)
    print(P)
    print(Q)
    print(omega)
    # print(P.shape,Q.shape,omega.shape)
    return P, Q, omega, cov_matrix


def generate_matrix(closes,number,price_pred,mse_record):
    # price_pred = np.array([83,30,13,12])
    # mse_record = np.array([0.1,0.15,0.13,0.12])

    closes = np.array(closes)
    cov_matrix = np.corrcoef(closes[1:])
    # print(cov_matrix)
    price_now = []
    for i in range(1,number):
        if closes[i][0]!=0:
            price_now.append(closes[i][0])
        else:
            flag = 0
            for j in range(1,len(closes[i])):
                if closes[i][j]!=0:
                    price_now.append(closes[i][j])
                    flag = 1
                    break
            if flag==0:
                price_now.append(0.0001)
    price_now = np.array(price_now)
    
    price_change = (price_pred-price_now)/price_now
    # price_change = np.array(price_pred)
    confid = (1/mse_record)/np.max(1/mse_record)
    print(price_change.shape,confid.shape)
    #print(price_change)
    #print(confid)
    Q = []
    P = []
    omega = []
    for i in range(number-1):
        Q.append(price_change[i])
        tmp = []
        tmp_o = []
        for j in range(number-1):
            if j==i:
                tmp.append(1)
                tmp_o.append(confid[i]*confid[0])
            else:
                tmp.append(0)
                tmp_o.append(0)
        P.append(tmp)
        omega.append(tmp_o)
    Q = np.array(Q).reshape(-1, 1)
    P = np.array(P)
    omega = np.array(omega)
    #print(P)
    #print(Q)
    #print(omega)
    print(P.shape,Q.shape,omega.shape)
    return P, Q, omega, cov_matrix

def generate_dict(etfs,closes,number,price_pred,mse_record):
    closes = np.array(closes)
    cov_matrix = np.corrcoef(closes[1:])
    # print(cov_matrix)
    price_now = []
    for i in range(1,number):
        if closes[i][0]!=0:
            price_now.append(closes[i][0])
        else:
            flag = 0
            for j in range(1,len(closes[i])):
                if closes[i][j]!=0:
                    price_now.append(closes[i][j])
                    flag = 1
                    break
            if flag==0:
                price_now.append(0.0001)
    price_now = np.array(price_now)
    
    price_change = (price_pred-price_now)/price_now
    # price_change = np.array(price_pred)
    confid = (1/mse_record)/np.max(1/mse_record)
    confidences_dict = np.array(confid)
    # confidences_dict =  {}
    views_dict ={}

    for i in range(len(etfs)):
        # confidences_dict[etfs[i]] = confid[i]
        views_dict[etfs[i]] = price_change[i]

    return views_dict,confidences_dict,cov_matrix

# %%
if __name__ == '__main__':

    db_name = 'my_etf'
    list_etf = ['TW_etf']
    (y,expect_reward,nnnn,month) = (2018,0.08,1,6) # 2018/1/1~2018/12/31
    # market_etf = 'SPY'
    market_etf = '0050.TW'
    number = 3
    # cluster = 'type'
    cluster = 'corr'

    print('generate data')
    closes,volumes,volatilitys,groups,number = generate_input_data.generate_data(y,expect_reward,nnnn,month,db_name,cluster,number,market_etf,list_etf)
    
    price_pred = np.array([83,30,13,12])
    mse_record = np.array([0.1,0.15,0.13,0.12])

    P, Q, omega, cov_matrix = generate_matrix(closes,number,price_pred,mse_record)
    
    print(P)
    print(Q)
    print(omega)
    print(cov_matrix)

    # date1 = datetime.date(y,month,1)
    # date2 = datetime.date(y,month+1,1)
    # weights = bl_weight.get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    # print(weights)



# %%
