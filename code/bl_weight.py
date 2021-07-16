# %%
from pypfopt import black_litterman, risk_models
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
import pymysql
import datetime

# %%
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
    bl = BlackLittermanModel(cov_matrix, Q=Q, P=P, omega=omega)
    market_prices = get_market_price(db_name,date1,date2,market_etf)
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    bl.bl_weights(delta)
    weights = bl.clean_weights()
    # print(weights)
    return weights

# %%
if __name__ == '__main__':
    db_name = 'my_etf'
    # list_etf = ['TW_etf']
    (y,expect_reward,nnnn,month) = (2018,0.08,1,6) # 2018/1/1~2018/12/31
    date1 = datetime.date(y,month,1)
    date2 = datetime.date(y,month+1,1)
    market_etf = 'SPY'

    Q = np.array([-0.20, 0.05, 0.10, 0.15]).reshape(-1, 1)
    P = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0]
        ]
    )
    cov_matrix = np.array(
        [
            [1,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1]
        ]
    )
    omega= np.array(
        [
            [0.4,   0,   0,   0],
            [  0, 0.3,   0,   0],
            [  0,   0, 0.5,   0],
            [  0,   0,   0, 0.6]
        ]
    )

    weights = get_bl_weight(cov_matrix,Q,P,omega,db_name,date1,date2,market_etf)
    print(weights)

# %%
