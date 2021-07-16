# %%
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statistics
from statsmodels.tsa.arima_model import ARIMA
import warnings
import time
from pmdarima.arima import auto_arima
warnings.filterwarnings("ignore")
# %%

def holtwinter(origin_data,filepath,filename):
    # data = origin_data
    mean = np.mean(origin_data)
    std = statistics.stdev(origin_data)
    # scaler = StandardScaler()
    # data = list(scaler.fit_transform(origin_data.reshape(-1, 1)))
    data_scaler = (origin_data-mean)/std
    data_min = np.min(data_scaler)
    data = list(data_scaler-data_min+0.001)
    data_sr = pd.Series(data[:231])
    # print(data_sr)
    fit1 = ExponentialSmoothing(data_sr, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)

    train_pred_y = list(fit1.fittedvalues)
    predict_y = list(fit1.forecast(21))
    l1, = plt.plot((train_pred_y + predict_y))
    l5, = plt.plot(data)
    plt.legend(handles = [l1, l5], labels = ["aa","data"], loc = 'best', prop={'size': 7})
    plt.savefig(filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()

    flag=0
    for i in range(len(predict_y)):
        y = predict_y[i]
        if np.isnan(y):
            index = i
            flag = 1
            break
    
    if flag ==0:
        # print('predict_y',predict_y)
        test_mse = np.sqrt( ( ( np.array(predict_y) - np.array(data[231:]) ) ** 2).mean() )
        print('HW-test mse =', test_mse)

        predict_close = (predict_y[-1]+ data_min - 0.001)*std + mean 
        # predict_close = predict_y[-1]
        print('HW-predict close =',predict_close)
    else:
        test_mse = np.sqrt( ( ( np.array(predict_y[:index]) - np.array(data[231:231+index]) ) ** 2).mean() )
        print('HW-test mse =', test_mse)

        predict_close = (predict_y[index-1]+ data_min - 0.001)*std + mean 
        # predict_close = predict_y[-1]
        print('HW-predict close =',predict_close)

    return test_mse,predict_close

# %%

def arima(origin_data,filepath,filename):
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(origin_data.reshape(-1, 1))
    data = pd.DataFrame(scaled_data,columns=['Close'])
    price = data.values
    length = 231
    train = list(price[:length])
    test =  list(price[length:])
    model = auto_arima(train,start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0,
                    D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, 
                    error_action='warn',trace=True,suppress_warnings=True,stepwise=True, random_state=20,n_fits=30)
    pred = model.predict(n_periods=21)
    # print(pred)
    zero = []
    for i in range(len(train)):
        zero.append(None)
    # date = data.index[length:len(price)]
    # predictions = []
    # low_bound = []
    # up_bound = []
    # real_data = []

    # for i in range(len(test)-21):
    #     model = ARIMA(train, order=(1, 1, 1))
    #     model_fit = model.fit(disp=0)
    #     pred = model_fit.forecast(steps=21)[0][20]
    #     predictions.append(pred)
    #     real = test[i]
    #     train.append(real[0]) 
    #     real = test[i+21]
    #     real_data.append(real[0]) 
    test_mse = np.sqrt( ( ( np.array(test) - np.array(pred) ) ** 2).mean() )
    print('ARIMA-test mse =', test_mse)

    # model = ARIMA(train, order=(1, 1, 1))
    # model_fit = model.fit(disp=0)
    predict_y = pred[20]
    # print(pred)
    mean = np.mean(origin_data)
    std = statistics.stdev(origin_data)
    # print(mean,std)
    predict_close = predict_y*std + mean
    print('ARIMA-predict close =',predict_close)

    plt.plot(train,label='train')
    plt.plot(zero+test,label='test')
    plt.plot(zero+list(pred),label='predict')
    plt.legend()
    # plt.show()
    plt.savefig(filepath+filename)
    plt.clf()
    plt.close()
    # plt.figure(figsize=(16, 7))
    # plt.plot(real_data, label='Original data')
    # # plt.plot(sup, label='Training data')
    # plt.plot(predictions, label='Testing data')
    # plt.legend()
    # plt.savefig(filepath+filename)
    # plt.clf()
    # plt.close()
    # fig = plt.figure(figsize = (10,5))
    # ax = fig.add_subplot()
    # fig.subplots_adjust(top=0.85)
    # ax.tick_params(labelsize=12)
    # ax.plot(real_data, color='red', label='Real', marker='o', markerfacecolor='red',markersize=6)
    # ax.plot(predictions, color='#121466', label='Pred', marker='o',markerfacecolor='#121466',markersize=6)
    # ax.legend()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    return test_mse,predict_close


# %%
if __name__ == '__main__':
    start = time.time()

    filepath = 'D:/Alia/Documents/asset allocation/experience/timeseries-us/'
    filename = 'testfig.jpg'

    origin_data = np.array( [77.1500015258789,77.25,76.8499984741211,76.9000015258789,77.69999694824219,78.55000305175781,79.44999694824219,79.55000305175781,79.75,79.69999694824219,81.4000015258789,81.30000305175781,80.55000305175781,80.6500015258789,80.4000015258789,80.44999694824219,80.0,80.75,80.4000015258789,79.94999694824219,80.3499984741211,81.55000305175781,81.55000305175781,82.19999694824219,82.05000305175781,82.19999694824219,82.4000015258789,82.69999694824219,82.6500015258789,82.0999984741211,82.30000305175781,82.44999694824219,82.1500015258789,83.0,82.0999984741211,81.3499984741211,81.80000305175781,82.5,82.19999694824219,82.3499984741211,82.94999694824219,82.9000015258789,82.1500015258789,81.5999984741211,81.25,80.75,81.30000305175781,81.0,81.6500015258789,81.1500015258789,81.05000305175781,81.8499984741211,81.75,82.55000305175781,82.69999694824219,82.55000305175781,82.30000305175781,82.69999694824219,82.94999694824219,83.0,82.8499984741211,82.94999694824219,82.5999984741211,82.25,82.69999694824219,82.55000305175781,82.8499984741211,82.30000305175781,82.3499984741211,82.3499984741211,83.1500015258789,82.75,82.19999694824219,82.69999694824219,81.5999984741211,80.94999694824219,80.30000305175781,80.69999694824219,80.4000015258789,80.6500015258789,81.5,81.5,81.5,81.8499984741211,82.0999984741211,82.0999984741211,82.0999984741211,83.4000015258789,84.19999694824219,84.05000305175781,84.5,84.30000305175781,84.5999984741211,84.9000015258789,84.55000305175781,84.5,84.4000015258789,84.3499984741211,84.25,84.30000305175781,84.94999694824219,85.19999694824219,85.0999984741211,84.80000305175781,84.69999694824219,84.8499984741211,85.30000305175781,84.94999694824219,84.4000015258789,84.55000305175781,84.30000305175781,84.1500015258789,83.30000305175781,83.5999984741211,84.3499984741211,84.0,84.75,85.1500015258789,85.19999694824219,85.1500015258789,84.1500015258789,83.5999984741211,83.69999694824219,82.25,82.5999984741211,82.9000015258789,82.25,80.8499984741211,80.9000015258789,80.9000015258789,81.5,80.94999694824219,81.0,81.75,81.0999984741211,81.19999694824219,80.80000305175781,81.05000305175781,81.0,81.30000305175781,81.44999694824219,80.6500015258789,80.9000015258789,81.69999694824219,82.1500015258789,82.5999984741211,83.3499984741211,83.5,83.75,84.0999984741211,84.1500015258789,83.75,83.4000015258789,84.0999984741211,84.6500015258789,85.0,85.1500015258789,86.30000305175781,87.1500015258789,87.94999694824219,88.30000305175781,87.19999694824219,87.44999694824219,87.5,85.55000305175781,84.55000305175781,84.6500015258789,85.25,84.9000015258789,83.5999984741211,79.6500015258789,80.6500015258789,80.5999984741211,79.05000305175781,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,81.5,81.0,82.0,82.4000015258789,82.3499984741211,82.3499984741211,81.75,81.1500015258789,80.75,82.1500015258789,81.80000305175781,82.75,82.8499984741211,84.1500015258789,84.94999694824219,84.30000305175781,84.25,83.9000015258789,83.8499984741211,83.80000305175781,83.8499984741211,83.55000305175781,82.0999984741211,82.19999694824219,83.4000015258789,82.25,82.0999984741211,82.8499984741211,82.25,81.5,81.5,81.5,81.5,82.19999694824219,82.5,82.8499984741211,82.5,82.5,82.30000305175781,81.4000015258789,81.5999984741211,82.6500015258789,80.75,79.94999694824219,79.55000305175781,79.30000305175781,79.05000305175781,79.19999694824219,80.0,80.0,79.4000015258789,78.55000305175781,78.94999694824219,79.5,80.0,80.44999694824219,80.6500015258789,81.5999984741211,82.5,81.6500015258789,81.75,81.19999694824219,80.94999694824219,82.05000305175781,81.75,81.3499984741211,81.55000305175781,81.8499984741211,82.19999694824219,81.8499984741211,80.4000015258789,80.75])

    # mse, predict_price = holtwinter(origin_data,filepath,filename)
    mse, predict_price = arima(origin_data,filepath,filename)

    end = time.time()
    print(end-start)
# %%
