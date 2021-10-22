# %%
from pypfopt import black_litterman, risk_models
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
import pymysql
import datetime
import price2matrix
import generate_input_data
import matlab.engine
# %%

def get_bl_weight_by_matlab(Q,P,omega):
    q_df = pd.DataFrame(Q)
    p_df = pd.DataFrame(P)
    o_df = pd.DataFrame(omega)
    q_df.to_csv('matrix_q.csv',index=False,header=False)
    p_df.to_csv('matrix_p.csv',index=False,header=False)
    o_df.to_csv('matrix_omega.csv',index=False,header=False)
    eng = matlab.engine.start_matlab()
    ans = eng.bl()
    weights = np.round(np.array(ans).reshape(len(ans)),5)
    return weights

def get_market_price(db_name,date1,date2,market_etf):
    # db_name = 'my_etf'
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()
    # sql = "select `date`,`SPY` from `close` where date > '2021-01-01' and date < '2021-02-01'"
    sql = "select `date`,`"+market_etf+"` from `close` where date > '"+str(date1)+"' and date < '"+str(date2) +"'"
    # print(sql)
    cursor.execute(sql)
    result_close = cursor.fetchall()
    market_prices_df = pd.DataFrame(list(result_close))
    market_prices = pd.Series(list(market_prices_df[1]),index=list(market_prices_df[0]))
    return market_prices

def get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf):
    #mcap_dict = get_market_cap_dict(etfs,db_name)
    market_prices = get_market_price(db_name,date1,date2,market_etf)
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    #market_prior = black_litterman.market_implied_prior_returns(mcap_dict, delta, cov_matrix)
    bl = BlackLittermanModel(cov_matrix,pi='equal', Q=Q, P=P, omega=omega)
    bl.bl_weights(delta)
    cleaned_weights = bl.clean_weights()
    #rets = bl.bl_returns()
    #print(rets)
    #ef = EfficientFrontier(rets, cov_matrix,weight_bounds=(0.05,0.5))
    # weights = ef.efficient_return(target_return=0.001,market_neutral=True)
    # ef.efficient_risk(0)
    # weights = ef.max_sharpe()
    #weights = ef.min_volatility()
    #cleaned_weights = ef.clean_weights()
    
    return cleaned_weights

def get_market_cap_dict(etfs,db_name):
    db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
    cursor = db.cursor()
    all_result = []
    for etf in etfs:
        sql = "select `name`,`規模` from `detail` where name = '" + etf + "'"
        cursor.execute(sql)
        result = cursor.fetchall()
        all_result.append(list(result[0]))
    market_cap_df = pd.DataFrame(list(all_result),columns=['name','market cap'])
    for i in range(len(market_cap_df)):
        scale = market_cap_df['market cap'][i]
        market_cap_df['market cap'][i] = float((scale.split('(')[0]).replace(',',''))*100*10000
    # print(market_cap_df)
    mcap_dict = {ticker : cap for ticker, cap in zip(market_cap_df['name'].values, market_cap_df['market cap'].values)}
    # print(mcap_dict)
    return mcap_dict

def get_bl_weight_new(cov_matrix,db_name,date1,date2,market_etf,etfs,views_dict,confidences_dict,prices):
    mcap_dict = get_market_cap_dict(etfs,db_name)
    # calculate asset covariance and delta
    # market-implied risk premium, which is the market’s excess return divided by its variance
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    market_prices = get_market_price(db_name,date1,date2,market_etf)
    delta = black_litterman.market_implied_risk_aversion(market_prices, risk_free_rate=0.05796)
    # print(mcap_dict)
    # print(delta)
    # print(S)
    # calculate prior - market implied retunrs
    market_prior = black_litterman.market_implied_prior_returns(mcap_dict, delta, S)
    #print(market_prior)
    bl = BlackLittermanModel(S, pi=market_prior,tau=0.5, absolute_views=views_dict,omega="idzorek",view_confidences=confidences_dict)
    bl.bl_weights(delta)
    #rets = bl.bl_returns()
    #print(rets)
    #ef = EfficientFrontier(rets, cov_matrix,weight_bounds=(-3,3))
    #weights = ef.min_volatility()
    #cleaned_weights = ef.clean_weights()
    cleaned_weights = bl.clean_weights()
    
    return cleaned_weights
# %%
def get_no_short_weights(origin_w):
    weight = []
    clean_w = []
    add = 0
    count1 = 0
    count2 = 0
    for w in origin_w:
        if w > 0.05 and w<0.5:
            add+=w
        elif w<=0.05:
            count1+=1
        else:
            count2+=1
    for w in origin_w:
        if w > 0.05 and w<0.5:
            tmp = w/add*(1-0.05*count1-0.5*count2)
        elif w<=0.05:
            tmp = 0.05
        else:
            tmp = 0.5
        weight.append(tmp)
    for w in weight:
        clean_w.append(round(w,5))
    if sum(clean_w)>1:
        clean_w[-1] -= (1-sum(clean_w))
    if sum(clean_w)<1:
        clean_w[-1] += (sum(clean_w)-1)
    #print(sum(clean_w))
    return clean_w

# %%
if __name__ == '__main__':
    db_name = 'my_etf'
    list_etf = ['US_etf']
    (y,expect_reward,nnnn,month) = (2015,0.08,1,1) # 2018/1/1~2018/12/31
    date1 = datetime.date(y,month,1)
    date2 = datetime.date(y,month+1,1)
    cluster = 'corr'
    market_etf = 'SPY'
    etfs = ['ITOT' ,'VEU' ,'VNQ' ,'AGG']
    groups = [['SPY'],['ITOT'] ,['VEU'] ,['VNQ'] ,['AGG']]
    number = len(groups)
    cov_matrix = np.array(
        [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]
    )

    closes,volumes,volatilitys,groups,number,every_close_finalday = generate_input_data.generate_data_d(y,nnnn,month,db_name,cluster,number,market_etf,list_etf,groups)
    price_pred = np.array([42,41,62,94])
    mse_record = np.array([0.1,0.15,0.13,0.12])
    views_dict,confidences_dict,cov_matrix = price2matrix.generate_dict(etfs,closes,number,price_pred,mse_record)
    price = pd.DataFrame(closes[1:]).T
    for i in range(len(closes)-1):
        price.rename(columns = {i:etfs[i]}, inplace = True)
    # print(price)
    # views_dict = {'ITOT': 0.03, 'VEU': 0.01, 'VNQ': 0.02, 'AGG': 0.04}
    # confidences_dict =  {'ITOT': 0.7, 'VEU': 0.6, 'VNQ': 0.8, 'AGG': 0.5}
    # confidences_dict = np.array([0.7,0.6,0.8,0.5])
    weights = get_bl_weight_new(cov_matrix,db_name,date1,date2,market_etf,etfs,views_dict,confidences_dict,price)
    

    # Q = np.array([-0.20, 0.05, 0.10, 0.15]).reshape(-1, 1)
    # P = np.array(
    #     [
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0]
    #     ]
    # )
    # cov_matrix = np.array(
    #     [
    #         [1,0,0,0],
    #         [0,1,0,0],
    #         [0,0,1,0],
    #         [0,0,0,1]
    #     ]
    # )
    # omega= np.array(
    #     [
    #         [0.4,   0,   0,   0],
    #         [  0, 0.3,   0,   0],
    #         [  0,   0, 0.5,   0],
    #         [  0,   0,   0, 0.6]
    #     ]
    # )

    # weights = get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    print(weights)

# %%
