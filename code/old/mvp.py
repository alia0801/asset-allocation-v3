# %%
from os import close
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
# %%
# tick = ['AMZN', 'AAPL', 'NFLX', 'XOM', 'T']
# price_data = web.get_data_yahoo(tick, start = '2014-01-01',end = '2018-05-31')['Adj Close']

closes = [[78.55000305175781,  78.94999694824219,  79.5,  80.0,  80.44999694824219,  80.6500015258789,  81.5999984741211,  82.5,  81.6500015258789,  81.75,  81.19999694824219,  80.94999694824219,  82.05000305175781,  81.75,  81.3499984741211,  81.55000305175781,  81.8499984741211,  82.19999694824219,  81.8499984741211,  80.4000015258789,  80.75],
 [29.749730032843512,  29.712702931584538,  29.992972928124505,  30.33756788356884,  30.445135039252204,  30.571621727299046,  30.84945923573262,  31.177567546432083,  31.003513426394075,  30.911351216805947,  30.78540530075898,  30.715945656235153,  31.090540447750605,  30.97270255475431,  30.700540336402685,  30.601351609101165,  30.638107969954206,  30.750270031593942,  30.51837830930143,  30.049729759628708,  30.334594520362646],
 [13.524615544539232,  13.517692272479717,  13.500000036679781,  13.416153871096098,  13.441538590651293,  13.409230819115272,  13.333846238943247,  13.219230835254375,  13.276923142946684,  13.351538511422964,  13.378461507650522,  13.383846282958984,  13.325384653531588,  13.376153835883507,  13.419230827918419,  13.433076968559853,  13.368461498847374,  13.261538468874418,  13.333846275623028,  13.456153979668251,  13.386923019702618],
 [12.032999992370605,  12.03100004196167,  11.97799997329712,  12.01200008392334,  11.978999996185303,  11.966000080108643,  12.00600004196167,  12.064999961853028,  12.05299997329712,  12.031999969482422,  12.011000156402588,  12.008999919891357,  12.026000213623046,  12.005000114440918,  12.002999973297118,  11.982000160217286,  12.014999961853027,  12.110000038146973,  11.973999881744385,  11.980999851226807,  11.95299997329712]]

price_data = pd.DataFrame(closes).T

log_ret = np.log(price_data/price_data.shift(1))

cov_mat = log_ret.cov() * 252
print(cov_mat)

# %%
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

# %%

for i in range(num_port):
    wts = np.random.uniform(size = len(price_data.columns))
    wts = wts/np.sum(wts)
    
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
# %%
names = price_data.columns
min_var = all_wts[port_risk.argmin()]
print(min_var)
max_sr = all_wts[sharpe_ratio.argmax()]
print(max_sr)
print(sharpe_ratio.max())
print(port_risk.min())
# %%
min_var = pd.Series(min_var, index=names)
min_var = min_var.sort_values()
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.set_xlabel('Asset')
ax1.set_ylabel("Weights")
ax1.set_title("Minimum Variance Portfolio weights")
min_var.plot(kind = 'bar')
plt.show()
# %%
max_sr = pd.Series(max_sr, index=names)
max_sr = max_sr.sort_values()
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.set_xlabel('Asset')
ax1.set_ylabel("Weights")
ax1.set_title("Tangency Portfolio weights")
max_sr.plot(kind = 'bar')
plt.show()
# %%
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.set_xlabel('Risk')
ax1.set_ylabel("Returns")
ax1.set_title("Portfolio optimization and Efficient Frontier")
plt.scatter(port_risk, port_returns)
plt.show()
# %%
