# %%
from pypfopt import black_litterman, risk_models
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
import pymysql

# %%
db_name = 'my_etf'
db = pymysql.connect(host="localhost", user="root", password="esfortest", database=str(db_name))
cursor = db.cursor()
sql = "select `date`,`SPY` from `close` where date > '2021-01-01'"
cursor.execute(sql)
result_close = cursor.fetchall()
market_prices_df = pd.DataFrame(list(result_close))
# market_prices_df.set_index(0 , inplace=True)
market_prices = pd.Series(list(market_prices_df[1]),index=list(market_prices_df[0]))

# %%

Q = np.array([-0.20, 0.05, 0.10, 0.15]).reshape(-1, 1)
P = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0]
    ]
)
# pi = np.array( [ [0.01], [0.02], [0.04], [0.03], [0.06], [0,1], [0.02], [0.07], [0.05], [0.06] ] )
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
# viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
# bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict, Q=Q, P=P)
bl = BlackLittermanModel(cov_matrix, Q=Q, P=P, omega=omega)
# %%
rets = bl.bl_returns()
ef = EfficientFrontier(rets, cov_matrix)


# %%
# OR use return-implied weights
delta = black_litterman.market_implied_risk_aversion(market_prices)
bl.bl_weights(delta)
weights = bl.clean_weights()
print(weights)
# %%
