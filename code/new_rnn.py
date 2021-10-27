# %%
import numpy as np
from numpy import random
import pandas as pd
from pandas.core.series import Series
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
import random
import statistics
import time
import gc
import scipy.stats as st
import tensorflow_probability as tfp
# from keras_self_attention import SeqSelfAttention
# %%

def window_data_new(data,data2,data3, window_size_x,window_size_y):
    X_tmp = []
    y = []
    i = 0
    while (i+window_size_x+window_size_y) < len(data):
        X_tmp.append( [ data[ i: i+window_size_x ], data2[ i: i+window_size_x ], data3[ i: i+window_size_x ]])
        y.append(data[ i+window_size_x:i+window_size_x+window_size_y ])     
        i += 1
    assert len(X_tmp) ==  len(y)
    X = []
    for i in range(len(X_tmp)):
      raw0 = X_tmp[i][0]
      raw1 = X_tmp[i][1]
      raw2 = X_tmp[i][2]
      raw = []
      for j in range(len(X_tmp[i][0])):
        # print([ raw0[j][0], raw1[j][0], raw2[j][0] ])
        raw.append( [ raw0[j][0], raw1[j][0], raw2[j][0] ] )
        # raw.append(  raw0[j]  )
      X.append(raw)
    return X, y

def window_data(data,data2,data3, window_size_x,window_size_y):
    X_tmp = []
    y = []
    i = 0
    while (i+window_size_x+window_size_y) < len(data):
        X_tmp.append( [ data[ i: i+window_size_x ], data2[ i: i+window_size_x ], data3[ i: i+window_size_x ]])
        y.append(data[ i+window_size_x+window_size_y ])     
        i += 1
    assert len(X_tmp) ==  len(y)
    X = []
    for i in range(len(X_tmp)):
      raw0 = X_tmp[i][0]
      raw1 = X_tmp[i][1]
      raw2 = X_tmp[i][2]
      raw = []
      for j in range(len(X_tmp[i][0])):
        # print([ raw0[j][0], raw1[j][0], raw2[j][0] ])
        raw.append( [ raw0[j][0], raw1[j][0], raw2[j][0] ] )
        # raw.append(  raw0[j]  )
      X.append(raw)
    return X, y

def window_data_corr(data,data2,data3,data4, window_size_x,window_size_y):
    X_tmp = []
    y = []
    y2=[]
    i = 0
    while (i+window_size_x+window_size_y) < len(data):
        X_tmp.append( [ data[ i: i+window_size_x ], data2[ i: i+window_size_x ], data3[ i: i+window_size_x ]])
        y.append(data[ i+window_size_x+window_size_y ])     
        y2.append(data4[ i+window_size_x+window_size_y ])     
        i += 1
    assert len(X_tmp) ==  len(y)
    X = []
    for i in range(len(X_tmp)):
      raw0 = X_tmp[i][0]
      raw1 = X_tmp[i][1]
      raw2 = X_tmp[i][2]
      raw = []
      for j in range(len(X_tmp[i][0])):
        # print([ raw0[j][0], raw1[j][0], raw2[j][0] ])
        raw.append( [ raw0[j][0], raw1[j][0], raw2[j][0] ] )
        # raw.append(  raw0[j]  )
      X.append(raw)
    return X, y, y2

def window_data_market(data,data2,data3,data4, window_size_x,window_size_y):
    X_tmp = []
    y = []
    y2=[]
    i = 0
    while (i+window_size_x+window_size_y) < len(data):
        X_tmp.append( [ data[ i: i+window_size_x ], data2[ i: i+window_size_x ], data3[ i: i+window_size_x ],  data4[ i+window_size_y: i+window_size_x+window_size_y ]])
        # X_tmp.append( [ data[ i: i+window_size_x ], data2[ i: i+window_size_x ], data3[ i: i+window_size_x ],  data4[ i: i+window_size_x ]])
        y.append(data[ i+window_size_x+window_size_y ])     
        y2.append(data4[ i+window_size_x+window_size_y ])     
        i += 1
    assert len(X_tmp) ==  len(y)
    X = []
    for i in range(len(X_tmp)):
      raw0 = X_tmp[i][0]
      raw1 = X_tmp[i][1]
      raw2 = X_tmp[i][2]
      raw = []
      for j in range(len(X_tmp[i][0])):
        # print([ raw0[j][0], raw1[j][0], raw2[j][0] ])
        raw.append( [ raw0[j][0], raw1[j][0], raw2[j][0] ] )
        # raw.append(  raw0[j]  )
      X.append(raw)
    return X, y, y2

def find_new_center(input_point,center,new_r):
    # print('input_point:',input_point)
    # print('center:', center)
    # print('new_r:',new_r)
    v = center - input_point
    unit_vec = v / (v**2).sum()**0.5
    # print('unit_vec:',unit_vec)
    new_vec = unit_vec * new_r
    new_center = input_point + new_vec
    # print('new_center:',new_center)
    return new_center

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

def ecm(input_point,clusters,radius,centers):
    # print(input_point)
    # learn_type = 1 #學(LSTM_cell)
    # learn_type = 0 #不學(LSTM_cell_nolearn)
    if len(clusters)==0: #學
        # print('first point')
        learn_type = 1
        clusters.append([input_point])
        radius.append(0)
        centers.append(input_point)
    else:
        r_min = 10000
        close_cluster = -1
        flag = -1
        for i in range(len(clusters)):
            dist = eucliDist(centers[i],input_point)
            if dist<r_min:
                r_min = dist
                close_cluster = i
                if dist<radius[i]:
                    flag = 1
        if flag != -1: #在某群半徑內 不學
            # print('in radius')
            learn_type = 0
            clusters[close_cluster].append(input_point)
        else:
            si = []
            for j in range(len(clusters)):
                s = eucliDist(centers[j],input_point) + radius[j]
                si.append(s)
            sia = min(si)
            if sia > 2*r_min: #自成一群 學
                # print('bigger than r_min')
                learn_type = 1
                clusters.append([input_point])
                radius.append(0)
                centers.append(input_point)
            else: #加入某群並改中心半徑 不學
                # print('smaller than r_min')
                learn_type = 0
                clusters[close_cluster].append(input_point)
                radius[close_cluster] = sia/2
                new_center = find_new_center(input_point,centers[close_cluster],sia/2)
                centers[close_cluster] = new_center

    return learn_type

    # return 1
    # return random.randint(0, 1)

def cal_learn_type(data): #(batchsz,21,3)
    types = []
    for points in data:  # batchsz次
        # ECM初始化
        clusters = []
        radius = []
        centers = []
        # points :(21,3) 21個點 分別計算各點是否學習
        tmp_list = []
        for p in points: # 21次
            # p: 1個3D的點
            t = ecm(p,clusters,radius,centers)
            tmp_list.append(t)
        types.append(tmp_list)
    # print('learn types:',types)
    return types

class MyLSTMCell(keras.layers.LSTMCell):
    def __init__(self,units,learn_type_all):
        super(MyLSTMCell, self).__init__(units)
        self.count = 0
        self.learn_type_all = learn_type_all
    
    def call(self, inputs, states, training=None):
        # learn_type = [0,1,1,0,1,0,0,1,1,1, 0,1,1,1,1,0,0,1,0,1]
        # tmp = []
        # for i in range(units):
        #     tmp.append(random.random())
        # o_bakup_list = [tmp]
        # o_bakup = tf.Variable(initial_value=o_bakup_list)# ,shape=(1, units)

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        learn_type = self.learn_type_all[self.count%21]
        if max(learn_type)==0:
            learn_type[0]=1
        # print(learn_type)
        self.count+=1
        if self.count%21==0:
            print('count',self.count)

        o = []
        c = []
        count_head0 = 0
        flag = 0

        for i in range(len(learn_type)):#batchsz
            learn = learn_type[i]
            # print(i,'learn_type:',learn)
            if learn == 1:
                # print(i,'learn')

                if self.implementation == 1:
                    # print('implementation is 1')
                    inputs_i = [inputs[i]]
                    inputs_f = [inputs[i]]
                    inputs_c = [inputs[i]]
                    inputs_o = [inputs[i]]

                    k_i, k_f, k_c, k_o = array_ops.split(
                        self.kernel, num_or_size_splits=4, axis=1)
                    x_i = backend.dot(inputs_i, k_i)
                    x_f = backend.dot(inputs_f, k_f)
                    x_c = backend.dot(inputs_c, k_c)
                    x_o = backend.dot(inputs_o, k_o)

                    if self.use_bias:
                        b_i, b_f, b_c, b_o = array_ops.split(
                            self.bias, num_or_size_splits=4, axis=0)
                        x_i = backend.bias_add(x_i, b_i)
                        x_f = backend.bias_add(x_f, b_f)
                        x_c = backend.bias_add(x_c, b_c)
                        x_o = backend.bias_add(x_o, b_o)

                    h_tm1_i = h_tm1[i]
                    h_tm1_f = h_tm1[i]
                    h_tm1_c = h_tm1[i]
                    h_tm1_o = h_tm1[i]

                    x = (x_i, x_f, x_c, x_o)
                    h_tm1[i] = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
                    c_i, o_i = self._compute_carry_and_output(x, [h_tm1[i]], [c_tm1[i]])

                else:
                    # print('implementation not 1')
                    z = backend.dot( tf.convert_to_tensor([inputs[i]]), self.kernel )
                    z += backend.dot( tf.convert_to_tensor([h_tm1[i]]), self.recurrent_kernel)
                    if self.use_bias:
                        z = backend.bias_add(z, self.bias)

                    z = array_ops.split(z, num_or_size_splits=4, axis=1)
                    c_i, o_i = self._compute_carry_and_output_fused(z, tf.convert_to_tensor([c_tm1[i]]))
                # print(o_i)
                c.append(c_i[0])
                # o_bakup = o_i
                # print('assign o_backup')
                if flag == 0:
                    flag = 1
                    for k in range(count_head0+1):
                        o.append(o_i[0])
                else:
                    o.append(o_i[0])

            else:
                # print(i,'no learn')
                if flag == 0:
                    # print('flag is 0')
                    count_head0+=1
                    # if count_head0==batchsz:
                    #     o_i = o_bakup
                    #     for k in range(count_head0):
                    #         o.append(o_i[0])
                else:
                    o.append(o_i[0])

                c_i = c_tm1[i]
                c.append(c_i)

        c = tf.convert_to_tensor(c)
        o = tf.convert_to_tensor(o)
        h = o * self.activation(c)
        # print('c',c)
        # print('o',o)
        # print('h',h)
        return h, [h, c]

class Attention_ECM_lstm(keras.Model):
    def __init__(self,units,learn_type_all):
        super(Attention_ECM_lstm, self).__init__()
        self.lstm_en = layers.RNN( MyLSTMCell(units, learn_type_all=learn_type_all), input_shape=(21,3), return_sequences=True,unroll=True)
        self.lstm_de = layers.SimpleRNN(units,dropout=0.2,unroll=True)
        # self.lstm_de = layers.RNN( MyLSTMCell(units, learn_type_all=learn_type_all), input_shape=(21,3), return_sequences=True,unroll=True)
        self.attention = layers.Attention()
        self.fc = layers.Dense(32)
        self.out = layers.Dense(1)
    def call(self, inputs, training=None):
      #   x = self.lstm(inputs)
        # x = self.mylstm(inputs)
        x = self.lstm_en(inputs)
        x = self.attention([x, x])
        x = self.lstm_de(x)
        x = self.fc(x)
        out = self.out(x)
        return out

class myECM_lstm(keras.Model):
    def __init__(self,units,learn_type_all):
        super(myECM_lstm, self).__init__()
        self.mylstm = Sequential([
            layers.RNN( MyLSTMCell(units, learn_type_all=learn_type_all), input_shape=(21,3), return_sequences=True,unroll=True),
            layers.RNN( MyLSTMCell(units, learn_type_all=learn_type_all), input_shape=(21,3), return_sequences=True,unroll=True) ])
            # layers.SimpleRNN(units,dropout=0.2,unroll=True)])
        self.fc = layers.Dense(32)
        self.out = layers.Dense(1)
    def call(self, inputs, training=None):
      #   x = self.lstm(inputs)
        x = self.mylstm(inputs)
        x = self.fc(x)
        out = self.out(x)
        return out

class trad_lstm(keras.Model):
    def __init__(self,units):
        super(trad_lstm, self).__init__()
        self.lstm = Sequential([
            layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True),
            # layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)])
            layers.SimpleRNN(units,dropout=0.2,unroll=True)])
        self.rnn = Sequential([
            layers.SimpleRNN(units,input_shape=(21,3),dropout=0.2,return_sequences=True,unroll=True),
            layers.SimpleRNN(units,dropout=0.2,unroll=True)])
        self.fc = layers.Dense(32)
        self.out = layers.Dense(1)
    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        x = self.fc(x)
        out = self.out(x)
        return out

class Attention_lstm(keras.Model):
    def __init__(self,units):
        super(Attention_lstm, self).__init__()
        self.lstm_en = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)
        # self.lstm_de = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)
        self.lstm_de = layers.SimpleRNN(units,dropout=0.2,unroll=True)
        self.attention = layers.Attention()
        self.fc = layers.Dense(32)
        self.out = layers.Dense(1)
    def call(self, inputs, training=None):
        x = self.lstm_en(inputs)
        # x = self.attention([inputs, inputs])
        # x = self.lstm_en(x)
        x = self.attention([x, x])
        x = self.lstm_de(x)
        # x = self.attention([x, x])
        x = self.fc(x)
        out = self.out(x)
        return out

class Attention_lstm_4feature(keras.Model):
    def __init__(self,units):
        super(Attention_lstm_4feature, self).__init__()
        self.lstm_en = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,4), return_sequences=True,unroll=True)
        # self.lstm_de = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)
        self.lstm_de = layers.SimpleRNN(units,dropout=0.2,unroll=True)
        self.attention = layers.Attention()
        self.fc = layers.Dense(32)
        self.out = layers.Dense(1)
    def call(self, inputs, training=None):
        x = self.lstm_en(inputs)
        # x = self.attention([inputs, inputs])
        # x = self.lstm_en(x)
        x = self.attention([x, x])
        x = self.lstm_de(x)
        # x = self.attention([x, x])
        x = self.fc(x)
        out = self.out(x)
        return out

class Attention_lstm_pred1m(keras.Model):
    def __init__(self,units):
        super(Attention_lstm_pred1m, self).__init__()
        self.lstm_en = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)
        # self.lstm_de = layers.RNN( keras.layers.LSTMCell(units,dropout=0.2), input_shape=(21,3), return_sequences=True,unroll=True)
        self.lstm_de = layers.SimpleRNN(units,dropout=0.2,unroll=True)
        self.attention = layers.Attention()
        self.fc = layers.Dense(32)
        self.out = layers.Dense(21)
    def call(self, inputs, training=None):
        x = self.lstm_en(inputs)
        # x = self.attention([inputs, inputs])
        # x = self.lstm_en(x)
        x = self.attention([x, x])
        x = self.lstm_de(x)
        # x = self.attention([x, x])
        x = self.fc(x)
        out = self.out(x)
        return out
# %%
def get_interval(y_train,pred_train):
    res = y_train - pred_train
    alpha = 0.05
    #print(res.shape)
    bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
    #print(bootstrap)
    q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)
    #print(q_bootstrap)
    lower_bound =  q_bootstrap[0].mean()
    upper_bound =  q_bootstrap[1].mean()
    #y_pred = model.predict(X_test)
    #y_lower = y_pred + q_bootstrap[0].mean()
    #y_upper = y_pred + q_bootstrap[1].mean()

    return lower_bound,upper_bound,res

# %%
def lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

    rnn_model = trad_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)


    y_pred_list_train = []
    y_pred_list_test = []
    upper_bound=[]
    lower_bound=[]
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    avg_train_error = np.mean( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) )
    print('avg train error:',avg_train_error)
    # error_std = np.std( np.sqrt( (np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) ) ** 2 ) )
    error_std = np.std( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) )
    error_str = 'avg train error ='+ str(avg_train_error)+', std ='+str(error_std)
    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)

    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+avg_train_error+error_std)
        lower_bound.append(y_pred_test[i][0]+avg_train_error-error_std)

    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  

    
    y_pred_ans = rnn_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)
    adj_close = ((predict_y-avg_train_error)*std+mean)

    print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(predict_close)
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,error_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, adj_close

# %%
def atten_lstm_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

    rnn_model = Attention_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)


    y_pred_list_train = []
    y_pred_list_test = []
    upper_bound=[]
    lower_bound=[]
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)

    
    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (y_pred_ans*std+mean)

    print("predict_price:",predict_close[0][-1])
    pred_str = "predict_price:"+str(predict_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]

# %%
def lstm_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

    rnn_model = trad_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)


    y_pred_list_train = []
    y_pred_list_test = []
    upper_bound=[]
    lower_bound=[]
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)

    
    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (y_pred_ans*std+mean)

    print("predict_price:",predict_close[0][-1])
    pred_str = "predict_price:"+str(predict_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]

# LSTM 直接輸出21天
def atten_lstm_out1m(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    # print(scaled_market.shape,scaled_data.shape)
    # corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    # print('corr2market=',corr2market)
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train = X[:120]
    y_train = y[:120]
    X_test = X[120:]
    y_test = y[120:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

    # def costum_loss(y_true,y_pred):
    #     loss1 = tf.reduce_mean(tf.square(y_true-y_pred))
    #     market_data = tf.convert_to_tensor(scaled_market[window_size_x:window_size_x+batchsz].astype(np.float32))
    #     # print(y_pred.shape) #(30,1)
    #     # print(market_data.shape) #(252,1)
    #     corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
    #     loss2 = abs(corr-corr2market)
    #     return 0.7*loss1 + 0.3*loss2

    rnn_model = Attention_lstm_pred1m(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=['mse'],metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)

    print(y_pred_train.shape)
    print(y_pred_test.shape)

    y_pred_list_train = []
    y_pred_list_test = []
    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][-1])
        y_pred_list_test.append(None)
        # upper_bound.append(None)
        # lower_bound.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][-1])


    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  

    y_pred_ans = rnn_model.predict(X_1m_run)
    predict_y = y_pred_ans[0]
    print(predict_y)

    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)

    return 0, 0
    # return test_mse, adj_close


# %%
# 下界+bl
def atten_lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    # print(scaled_market.shape,scaled_data.shape)
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

#    def costum_loss(y_true,y_pred):
#        loss1 = tf.reduce_mean(tf.square(y_true-y_pred))
#        market_data = tf.convert_to_tensor(scaled_market[window_size_x:window_size_x+batchsz].astype(np.float32))
#        # print(y_pred.shape) #(30,1)
#        # print(market_data.shape) #(252,1)
#        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
#        loss2 = abs(corr-corr2market)
#        return 0.7*loss1 + 0.3*loss2

    rnn_model = Attention_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss='mse',metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)


    y_pred_list_train = []
    y_pred_list_test = []
    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
        # upper_bound.append(None)
        # lower_bound.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])


    # train_mse = np.sqrt( ( ( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) ) ** 2).mean() )
    # print(np.array(y_pred_list_train))
    # print(np.array(y[:len(y_pred_list_train)]))
    # avg_train_error = np.mean( np.sqrt( (np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) ) ** 2 ) )
    avg_train_error = np.mean( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) )
    print('avg train error:',avg_train_error)
    # error_std = np.std( np.sqrt( (np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) ) ** 2 ) )
    error_std = np.std( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) )
    error_str = 'avg train error ='+ str(avg_train_error)+', std ='+str(error_std)
    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)

    # st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
    
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+avg_train_error+error_std)
        lower_bound.append(y_pred_test[i][0]+avg_train_error-error_std)
        

    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  

    
    y_pred_ans = rnn_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)
    adj_close = ((predict_y)*std+mean)
    # adj_close = ((predict_y-avg_train_error)*std+mean)

    print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(predict_close)
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    # plt.plot(upper_bound, label='Test upper bound')
    # plt.plot(lower_bound, label='Test lower bound')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,error_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, adj_close
    
# %%
# 下界+loss+bl
def atten_lstm_lowb_loss_bl(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)

    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  

    
    y_pred_ans = rnn_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)
    adj_close = ((predict_y+low_b)*std+mean)

    print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(adj_close)
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, adj_close

# 下界+loss+blNewVeiw
def atten_lstm_lowb_loss_blNewVeiw(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)

    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  

    
    y_pred_ans = rnn_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    # predict_close = (predict_y*std+mean)
    # adj_close = ((predict_y+low_b)*std+mean)
    prob,prices = get_interval_probibility(low_b,up_b,res,mean,std,predict_y)

    # print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(prices[1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    
    return prob,prices

def get_interval_probibility(low_b,up_b,res,mean,std,price):

    h_price = price + up_b
    l_price = price + low_b

    count_m = 0
    count_l = 0
    count_h = 0

    for i in res:
        if i < up_b+0.1 and i > up_b-0.1:
            count_h+=1
        elif i < low_b+0.1 and i > low_b-0.1:
            count_l+=1
        elif i < 0.1 and i > -0.1:
            count_m+=1
    h_prob = count_h/len(res)
    m_prob = count_m/len(res)
    l_prob = count_l/len(res)

    h_price = (h_price*std+mean)
    m_price = (price*std+mean)
    l_price = (l_price*std+mean)

    prob = [h_prob,m_prob,l_prob]
    prices = [h_price,m_price,l_price]

    # print(prob,prices)

    return prob,prices

# %%
# 下界+loss+mvp
def atten_lstm_lowb_loss_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    # def costum_loss(y_true,y_pred,cut):
    #     loss1 = tf.reduce_mean(tf.square(y_true-y_pred))
    #     market_data = tf.convert_to_tensor(y_market[batchsz*cut:batchsz*(cut+1)].astype(np.float32))
    #     print(y_pred.shape) #(30,1)
    #     print(market_data.shape) #(252,1)
    #     corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
    #     loss2 = abs(corr-corr2market)
    #     return 0.7*loss1 + 0.3*loss2

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)
        

    print('start predict...')
    # scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    # scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    # scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    # X_1m_run = []
    # for i in range(batchsz):
    #     X_1m_run.append(X_1m)
    # X_1m_run = np.array(X_1m_run)  

    # final_pred_x = np.array(X[-21:])

    
    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(y_pred_ans)
    # predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans+low_b)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]

# %%
# 下界+lossAll+mvp
def atten_lstm_lowb_lossAll_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')

    def innerLoss(y_true,y_pred):
        tmp = round(len(scaled_market)/batchsz)
        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        # y_pred_list = y_pred.numpy().to_list()
        # y_pred_list_all = y_pred_list*(round(len(scaled_market)/batchsz))+y_pred_list[:len(scaled_market)%batchsz]
        market_data = tf.convert_to_tensor(np.array(scaled_market[tmp*(-90):]).astype(np.float32))
        y_pred_data = tf.tile(y_pred,[tmp,1])
        # y_pred_data = tf.concat(0,[y_pred_data,y_pred[:36]])
        corr = tfp.stats.correlation(market_data, y_pred_data, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss():
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred)
        return inner


    rnn_model = Attention_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(),metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)
    #print(y_pred_train.shape,y_train.shape)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    #print(pred_train.shape,correct_y.shape)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)
        

    print('start predict...')
    # scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    # scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    # scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    y_pred_ans = rnn_model.predict(final_pred_x)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans+low_b)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]


# %%
# loss+mvp
def atten_lstm_loss_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)

    print('start predict...')
    # scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    # scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    # scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    # X_1m_run = []
    # for i in range(batchsz):
    #     X_1m_run.append(X_1m)
    # X_1m_run = np.array(X_1m_run)  

    # final_pred_x = np.array(X[-21:])

    
    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(y_pred_ans)
    # predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    #plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    #plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]
# %%
# 下界+loss+mvp
def atten_lstm_lowb_loss_market_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_market(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm_4feature(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)
        

    print('start predict...')
    # scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    # scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    # scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    # X_1m_run = []
    # for i in range(batchsz):
    #     X_1m_run.append(X_1m)
    # X_1m_run = np.array(X_1m_run)  

    # final_pred_x = np.array(X[-21:])

    
    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(y_pred_ans)
    # predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans+low_b)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_market, label='Market data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, adj_close.reshape(1, -1).tolist()[0]

# %%
# 下界+loss+mvp
def atten_lstm_oldlowb_loss_market_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    corr2market = Series(scaled_data.flatten()).corr(Series(scaled_market.flatten()))
    print('corr2market=',corr2market)
    X, y, y_market = window_data_market(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[540:]
        y_test += y[540:]
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')

    def innerLoss(y_true,y_pred, cut):

        loss1= tf.reduce_mean(tf.square(y_true-y_pred))
        market_data = tf.convert_to_tensor(np.array(y_market[batchsz*cut:batchsz*(cut+1)]).astype(np.float32))
        corr = tfp.stats.correlation(market_data, y_pred, sample_axis=0, event_axis=None)
        loss2 = abs(corr-corr2market)
        return 0.5*loss1 + 0.5*loss2

    def costum_loss(cut):
        def inner(y_true, y_pred):
            return innerLoss(y_true,y_pred, cut)
        return inner

    for cut in range(0,int(540/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        rnn_model = Attention_lstm_4feature(units)
        rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
        rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)

        if cut==0:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss=costum_loss(cut=cut),metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            models.append(rnn_model)
        else:
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            models.append(rnn_model)
        print('len(models):',len(models))

    y_pred_train_all=[]
    print('training!!!')
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    rnn_model = models[0]
    rnn_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
    y_pred_test = rnn_model.predict(X_test)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    # low_b,up_b,res = get_interval(correct_y,pred_train)
    avg_train_error = np.mean( np.array(y_pred_list_train) - np.array(y[:len(y_pred_list_train)]) )
    print('avg train error:',avg_train_error)
    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    # bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    # print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+avg_train_error)
        lower_bound.append(y_pred_test[i][0]-avg_train_error)
        

    print('start predict...')

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    
    y_pred_ans = rnn_model.predict(final_pred_x)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans-avg_train_error)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_market, label='Market data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    # plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, adj_close.reshape(1, -1).tolist()[0]



# %%
# 下界+mvp
def atten_lstm_lowb_mvp(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    scaled_market = scaler.fit_transform(data_to_use[3].reshape(-1, 1))
    
    X, y, y_market = window_data_corr(scaled_data, scaled_volume, scaled_volat,scaled_market, window_size_x,window_size_y)

    X_train = X[:540]
    y_train = y[:540]
    X_test = X[540:]
    y_test = y[540:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)

    data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    data = data.batch(batchsz,drop_remainder=True)
    data_iter = iter(data)
    samples = next(data_iter)
    
    print('create train model & training!!!')


    rnn_model = Attention_lstm(units)
    rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss='mse',metrics=['mse'])
    rnn_model.fit(data,epochs=epochs, validation_data = data_test,shuffle=True)

    y_pred_train = rnn_model.predict(X_train)
    y_pred_test = rnn_model.predict(X_test)
    #print(y_pred_train.shape,y_train.shape)

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train)):
        y_pred_list_train.append(y_pred_train[i][0])
        y_pred_list_test.append(None)
    for i in range(len(y_pred_test)):
        y_pred_list_test.append(y_pred_test[i][0])

    pred_train = np.array(y_pred_list_train)
    correct_y = np.array(y[:len(y_pred_list_train)]).reshape(-1)
    #print(pred_train.shape,correct_y.shape)
    low_b,up_b,res = get_interval(correct_y,pred_train)

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
    bound_str = 'upper bound = '+str(round(up_b,5))+', lower bound = '+str(round(low_b,5))
    print(bound_str)

    upper_bound = []
    lower_bound = []
    for i in range(len(y_pred_test)):
        upper_bound.append(y_pred_test[i][0]+up_b)
        lower_bound.append(y_pred_test[i][0]+low_b)
        

    print('start predict...')
    # scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    # scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    # scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_data[:-(window_size_x+window_size_y)],scaled_volume[:-(window_size_x+window_size_y)],scaled_volat[:-(window_size_x+window_size_y)]]
    final_pred_x = []

    for j in range(21):
        X_1m=[]
        for i in range(window_size_x):
            raw0 = X_tmp[0][i+j]
            raw1 = X_tmp[1][i+j]
            raw2 = X_tmp[2][i+j]
            raw = [ raw0[0], raw1[0], raw2[0]]
            X_1m.append(raw)
        final_pred_x.append(X_1m)
    final_pred_x = np.array(final_pred_x)
    print('final_pred_x.shape',final_pred_x.shape)

    # X_1m_run = []
    # for i in range(batchsz):
    #     X_1m_run.append(X_1m)
    # X_1m_run = np.array(X_1m_run)  

    # final_pred_x = np.array(X[-21:])

    
    y_pred_ans = rnn_model.predict(final_pred_x)
    # print(y_pred_ans)
    # predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del rnn_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_to_use[0])
    std = statistics.stdev(data_to_use[0])
    predict_close = (y_pred_ans*std+mean)
    adj_close = ((y_pred_ans+low_b)*std+mean)
    print(adj_close)

    print("predict_price:",adj_close[0][-1])
    pred_str = "predict_price:"+str(adj_close[0][-1])
    
    print('plot...')
    fig_y_min = -5
    fig_y_max = 5
    test_x = np.linspace(len(y_pred_list_train),len(y_pred_list_test)-1,num=(len(y_pred_list_test)-len(y_pred_list_train)))
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.fill_between(test_x,upper_bound,lower_bound,color='pink')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,4.5,mse_str1)
    plt.text(0,4,mse_str2)
    plt.text(0,3.5,pred_str)
    plt.text(0,3,bound_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close.reshape(1, -1).tolist()[0]

# %%
def ecm_lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):
    
    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    learn_type_all_list = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[150:]
        y_test += y[150:]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')
    # model_path = 'D:/Alia/Documents/asset allocation/asset-allocation-v3/models/'

    for cut in range(0,int(150/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)
        learn_type_tmp = cal_learn_type(X_train) #(batchsz,21)
        learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        learn_type_all_list.append(learn_type_all)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        if cut==0:
            rnn_model = myECM_lstm(units,learn_type_all)
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            # rnn_model.save(model_path+'model-'+str(cut)+'.h5',save_format='tf')
            # tf.saved_model.save(rnn_model,model_path)
            # models.append('model-'+str(cut)+'.h5')
            models.append(rnn_model)
        else:
            rnn_model2 = myECM_lstm(units,learn_type_all)
            rnn_model2.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model2.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            # rnn_model2.save(model_path+'model-'+str(cut)+'.h5',save_format='tf')
            # tf.saved_model.save(rnn_model2,model_path+'model-'+str(cut)+'.h5')
            # models.append('model-'+str(cut)+'.h5')
            models.append(rnn_model2)
        print('len(models):',len(models))

    
    print('training!!!')
    y_pred_train_all = []
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            learn_type_all = learn_type_all_list[model_count] #(21,batchsz)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                # print(X_train.shape)
                # if len(X_train)>30:
                    # y_pred_train = 
                    # for j in range(ceil(len(X_train)/30)):
                    #     tmp = model.predict(X_train[j*30:(j+1)*30])
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    del models
    gc.collect()

    
    
    print('create test model...')
    y_pred_test_all = []
    for i in range(int(len(X_test)/batchsz)):
        print('model',i)
    
        X_test_1batch = X_test[i*batchsz:(i+1)*batchsz]
        learn_type_tmp = cal_learn_type(X_test_1batch) #(batchsz,21)
        learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)

        test_model = myECM_lstm(units,learn_type_all)
        test_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
        test_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
        test_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
        y_pred_test = test_model.predict(X_test_1batch)
        y_pred_test_all.append(y_pred_test)

        del test_model
        gc.collect()

    

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test_all)):
        for j in range(len(y_pred_test_all[0])):
            y_pred_list_test.append(y_pred_test_all[i][j][0])
    # for i in range(len(y_pred_test)):
        # y_pred_list_test.append(y_pred_test[i][0])

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
       
    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  
    # X_1m_run = X_1m_run.reshape(X_test.shape)

    learn_type_tmp = cal_learn_type(X_1m_run) #(batchsz,21)
    learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)
    
    print('create predict model...')
    pred_model = myECM_lstm(units,learn_type_all)
    pred_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
    pred_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
    pred_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')

    # X_1m_run = X_1m_run.reshape(X_test.shape)
    y_pred_ans = pred_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del pred_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)[0]

    print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(predict_close)

    print('plot...')
    fig_y_min = -3
    fig_y_max = 3
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,2.7,mse_str1)
    plt.text(0,2.5,mse_str2)
    plt.text(0,2.3,pred_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close

# %%
def atten_ecm_lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,filename,lstm_filepath):

    print('prepare data...')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_use[0].reshape(-1, 1))
    scaled_volume = scaler.fit_transform(data_to_use[1].reshape(-1, 1))
    scaled_volat = scaler.fit_transform(data_to_use[2].reshape(-1, 1))
    X, y = window_data(scaled_data, scaled_volume, scaled_volat, window_size_x,window_size_y)

    X_train_all = []
    y_train_all = []
    models = []
    learn_type_all_list = []
    X_test = []
    y_test = []

    while len(X_test) < batchsz:
        X_test += X[150:]
        y_test += y[150:]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)

    data_test = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    data_test = data_test.batch(batchsz,drop_remainder=True)
    
    print('create train model...')
    # model_path = 'D:/Alia/Documents/asset allocation/asset-allocation-v3/models/'

    for cut in range(0,int(150/batchsz)):
        X_train = np.array(X[cut*batchsz:(cut+1)*batchsz]) #(batchsz,21,3)
        y_train = np.array(y[cut*batchsz:(cut+1)*batchsz]) #(batchsz,1)
        learn_type_tmp = cal_learn_type(X_train) #(batchsz,21)
        learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        learn_type_all_list.append(learn_type_all)
        data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        data = data.batch(batchsz,drop_remainder=True)
        data_iter = iter(data)
        samples = next(data_iter)

        if cut==0:
            rnn_model = Attention_ECM_lstm(units,learn_type_all)
            rnn_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            rnn_model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            # rnn_model.save(model_path+'model-'+str(cut)+'.h5',save_format='tf')
            # tf.saved_model.save(rnn_model,model_path)
            # models.append('model-'+str(cut)+'.h5')
            models.append(rnn_model)
        else:
            rnn_model2 = Attention_ECM_lstm(units,learn_type_all)
            rnn_model2.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
            rnn_model2.fit(data,epochs=1, validation_data = data_test,shuffle=True)
            # rnn_model2.save(model_path+'model-'+str(cut)+'.h5',save_format='tf')
            # tf.saved_model.save(rnn_model2,model_path+'model-'+str(cut)+'.h5')
            # models.append('model-'+str(cut)+'.h5')
            models.append(rnn_model2)
        print('len(models):',len(models))

    
    print('training!!!')
    y_pred_train_all = []
    for i in range(epochs+1):
        print("---------------------epoch:",i,"---------------------------")
        model_count = 0
        for model in models:
            # print(model_count)
            # model = tf.keras.models.load_model(model_path+model_name)
            X_train = X_train_all[model_count] #(batchsz,21,3)
            y_train = y_train_all[model_count] #(batchsz,1)
            learn_type_all = learn_type_all_list[model_count] #(21,batchsz)
            data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            data = data.batch(batchsz,drop_remainder=True)
            data_iter = iter(data)
            samples = next(data_iter)
            if i < epochs: #train
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
                model.save_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
            else: #predict train data
                if model_count==0:
                    print('calculate performance...')
                print('model',model_count)
                model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
                # print(X_train.shape)
                # if len(X_train)>30:
                    # y_pred_train = 
                    # for j in range(ceil(len(X_train)/30)):
                    #     tmp = model.predict(X_train[j*30:(j+1)*30])
                y_pred_train = model.predict(X_train)
                y_pred_train_all.append(y_pred_train)
            model_count+=1

    del models
    gc.collect()

    
    
    print('create test model...')
    y_pred_test_all = []
    for i in range(int(len(X_test)/batchsz)):
        print('model',i)
    
        X_test_1batch = X_test[i*batchsz:(i+1)*batchsz]
        learn_type_tmp = cal_learn_type(X_test_1batch) #(batchsz,21)
        learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)

        test_model = Attention_ECM_lstm(units,learn_type_all)
        test_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
        test_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
        test_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')
        y_pred_test = test_model.predict(X_test_1batch)
        y_pred_test_all.append(y_pred_test)

        del test_model
        gc.collect()

    

    y_pred_list_train = []
    y_pred_list_test = []
    for i in range(len(y_pred_train_all)):
        for j in range(len(y_pred_train_all[0])):
            y_pred_list_train.append(y_pred_train_all[i][j][0])
            y_pred_list_test.append(None)
    for i in range(len(y_pred_test_all)):
        for j in range(len(y_pred_test_all[0])):
            y_pred_list_test.append(y_pred_test_all[i][j][0])
    # for i in range(len(y_pred_test)):
        # y_pred_list_test.append(y_pred_test[i][0])

    y_model_cal = y_pred_list_train+y_pred_list_test[len(y_pred_list_train):]
    y_cal_mse = y[:len(y_model_cal)]
    total_mse = np.sqrt( ( ( np.array(y_model_cal) - np.array(y_cal_mse) ) ** 2).mean() )
    mse_str1 = 'total mse ='+ str(total_mse)
    print('total mse =', total_mse)
    test_mse = np.sqrt( ( ( np.array( y_model_cal[len(y_pred_list_train):] ) - np.array( y_cal_mse[len(y_pred_list_train):] ) ) ** 2).mean() )
    print('test mse =', test_mse)
    mse_str2 = 'test mse ='+ str(test_mse)
       
    print('start predict...')
    scaled_close_1m = scaler.fit_transform(data_a_month[0].reshape(-1, 1))
    scaled_volume_1m = scaler.fit_transform(data_a_month[1].reshape(-1, 1))
    scaled_vola_1m = scaler.fit_transform(data_a_month[2].reshape(-1, 1))

    X_tmp = [scaled_close_1m,scaled_volume_1m,scaled_vola_1m]
    X_1m = []
    for i in range(len(scaled_close_1m)):
        raw0 = X_tmp[0][i]
        raw1 = X_tmp[1][i]
        raw2 = X_tmp[2][i]
        raw = [ raw0[0], raw1[0], raw2[0]]
        X_1m.append(raw)
    X_1m = np.array(X_1m)

    X_1m_run = []
    for i in range(batchsz):
        X_1m_run.append(X_1m)
    X_1m_run = np.array(X_1m_run)  
    # X_1m_run = X_1m_run.reshape(X_test.shape)

    learn_type_tmp = cal_learn_type(X_1m_run) #(batchsz,21)
    learn_type_all = np.array(learn_type_tmp).T #(21,batchsz)
    
    print('create predict model...')
    pred_model = Attention_ECM_lstm(units,learn_type_all)
    pred_model.compile(optimizer  =keras.optimizers.Adam(),loss="mse",metrics=['mse'])
    pred_model.fit(data,epochs=1, validation_data = data_test,shuffle=True)
    pred_model.load_weights('D:/Alia/Documents/asset allocation/asset-allocation-v3/models/weights.h5')

    # X_1m_run = X_1m_run.reshape(X_test.shape)
    y_pred_ans = pred_model.predict(X_1m_run)
    predict_y = y_pred_ans[0][0]
    # print(predict_y)

    del pred_model
    gc.collect()

    # 轉回原來數值
    # y = (x – 平均值) / 標準偏差--> x =y*stdev+mean
    # mean = np.sum(data_a_month[0])/len(data_a_month[0])
    mean = np.mean(data_a_month[0])
    std = statistics.stdev(data_a_month[0])
    predict_close = (predict_y*std+mean)

    print("predict_price:",predict_close)
    pred_str = "predict_price:"+str(predict_close)

    print('plot...')
    fig_y_min = -3
    fig_y_max = 3
    plt.plot(y, label='Original data')
    plt.plot(y_pred_list_train, label='Training data')
    plt.plot(y_pred_list_test, label='Testing data')
    plt.ylim([fig_y_min,fig_y_max])
    plt.text(0,2.7,mse_str1)
    plt.text(0,2.5,mse_str2)
    plt.text(0,2.3,pred_str)
    plt.legend(loc="lower right")
    plt.savefig(lstm_filepath+filename)
    # plt.show()
    plt.clf()
    plt.close()
    
    return test_mse, predict_close

# %%
if __name__ == '__main__':
    start = time.time()
    batchsz = 30
    units = 64
    epochs = 10
    # window_size = 21
    window_size_x=21
    window_size_y=21
    data_to_use = np.array( [
        [77.1500015258789,77.25,76.8499984741211,76.9000015258789,77.69999694824219,78.55000305175781,79.44999694824219,79.55000305175781,79.75,79.69999694824219,81.4000015258789,81.30000305175781,80.55000305175781,80.6500015258789,80.4000015258789,80.44999694824219,80.0,80.75,80.4000015258789,79.94999694824219,80.3499984741211,81.55000305175781,81.55000305175781,82.19999694824219,82.05000305175781,82.19999694824219,82.4000015258789,82.69999694824219,82.6500015258789,82.0999984741211,82.30000305175781,82.44999694824219,82.1500015258789,83.0,82.0999984741211,81.3499984741211,81.80000305175781,82.5,82.19999694824219,82.3499984741211,82.94999694824219,82.9000015258789,82.1500015258789,81.5999984741211,81.25,80.75,81.30000305175781,81.0,81.6500015258789,81.1500015258789,81.05000305175781,81.8499984741211,81.75,82.55000305175781,82.69999694824219,82.55000305175781,82.30000305175781,82.69999694824219,82.94999694824219,83.0,82.8499984741211,82.94999694824219,82.5999984741211,82.25,82.69999694824219,82.55000305175781,82.8499984741211,82.30000305175781,82.3499984741211,82.3499984741211,83.1500015258789,82.75,82.19999694824219,82.69999694824219,81.5999984741211,80.94999694824219,80.30000305175781,80.69999694824219,80.4000015258789,80.6500015258789,81.5,81.5,81.5,81.8499984741211,82.0999984741211,82.0999984741211,82.0999984741211,83.4000015258789,84.19999694824219,84.05000305175781,84.5,84.30000305175781,84.5999984741211,84.9000015258789,84.55000305175781,84.5,84.4000015258789,84.3499984741211,84.25,84.30000305175781,84.94999694824219,85.19999694824219,85.0999984741211,84.80000305175781,84.69999694824219,84.8499984741211,85.30000305175781,84.94999694824219,84.4000015258789,84.55000305175781,84.30000305175781,84.1500015258789,83.30000305175781,83.5999984741211,84.3499984741211,84.0,84.75,85.1500015258789,85.19999694824219,85.1500015258789,84.1500015258789,83.5999984741211,83.69999694824219,82.25,82.5999984741211,82.9000015258789,82.25,80.8499984741211,80.9000015258789,80.9000015258789,81.5,80.94999694824219,81.0,81.75,81.0999984741211,81.19999694824219,80.80000305175781,81.05000305175781,81.0,81.30000305175781,81.44999694824219,80.6500015258789,80.9000015258789,81.69999694824219,82.1500015258789,82.5999984741211,83.3499984741211,83.5,83.75,84.0999984741211,84.1500015258789,83.75,83.4000015258789,84.0999984741211,84.6500015258789,85.0,85.1500015258789,86.30000305175781,87.1500015258789,87.94999694824219,88.30000305175781,87.19999694824219,87.44999694824219,87.5,85.55000305175781,84.55000305175781,84.6500015258789,85.25,84.9000015258789,83.5999984741211,79.6500015258789,80.6500015258789,80.5999984741211,79.05000305175781,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,81.5,81.0,82.0,82.4000015258789,82.3499984741211,82.3499984741211,81.75,81.1500015258789,80.75,82.1500015258789,81.80000305175781,82.75,82.8499984741211,84.1500015258789,84.94999694824219,84.30000305175781,84.25,83.9000015258789,83.8499984741211,83.80000305175781,83.8499984741211,83.55000305175781,82.0999984741211,82.19999694824219,83.4000015258789,82.25,82.0999984741211,82.8499984741211,82.25,81.5,81.5,81.5,81.5,82.19999694824219,82.5,82.8499984741211,82.5,82.5,82.30000305175781,81.4000015258789,81.5999984741211,82.6500015258789,80.75,79.94999694824219,79.55000305175781,79.30000305175781,79.05000305175781,79.19999694824219,80.0,80.0,79.4000015258789,78.55000305175781,78.94999694824219,79.5,80.0,80.44999694824219,80.6500015258789,81.5999984741211,82.5,81.6500015258789,81.75,81.19999694824219,80.94999694824219,82.05000305175781,81.75,81.3499984741211,81.55000305175781,81.8499984741211,82.19999694824219,81.8499984741211,80.4000015258789,80.75],
        [5000.0,7400.0,5100.0,16400.0,13700.0,12400.0,39300.0,31400.0,66300.0,37700.0,49800.0,42200.0,13000.0,41600.0,43900.0,13300.0,13300.0,16300.0,12800.0,70300.0,16800.0,17400.0,15500.0,20100.0,18900.0,10200.0,20000.0,7000.0,13100.0,9100.0,11900.0,12600.0,15900.0,11600.0,10200.0,14600.0,11200.0,11700.0,17200.0,42600.0,22900.0,11600.0,9500.0,7600.0,8900.0,11800.0,7700.0,17800.0,37400.0,6800.0,3200.0,9800.0,10500.0,4600.0,12700.0,41200.0,24100.0,38300.0,70500.0,87500.0,87500.0,53600.0,24400.0,23400.0,11100.0,17400.0,29100.0,35400.0,18100.0,17800.0,39100.0,30800.0,27800.0,17100.0,13700.0,21500.0,13500.0,29800.0,30300.0,42100.0,35600.0,37500.0,37500.0,19900.0,29100.0,43200.0,47700.0,34300.0,36000.0,30900.0,27100.0,56900.0,35800.0,45800.0,25300.0,26100.0,48100.0,52100.0,74500.0,126400.0,109600.0,62500.0,42700.0,35600.0,26000.0,38800.0,42700.0,91400.0,115800.0,12800.0,28200.0,73500.0,42100.0,26300.0,33500.0,37500.0,70600.0,74200.0,74200.0,39900.0,38300.0,26300.0,34400.0,29900.0,37200.0,44300.0,18900.0,26800.0,27900.0,49000.0,58000.0,48300.0,38500.0,29100.0,58600.0,52100.0,41200.0,39000.0,28100.0,21100.0,21100.0,17800.0,25200.0,25000.0,70700.0,82200.0,65500.0,30100.0,123100.0,75100.0,52800.0,31500.0,51100.0,95100.0,95100.0,94200.0,78700.0,49300.0,77900.0,235000.0,246000.0,170300.0,167400.0,132300.0,242600.0,148700.0,255600.0,57200.0,118700.0,105000.0,131900.0,123800.0,85400.0,98800.0,89500.0,53200.0,109400.0,83100.0,87800.0,44000.0,107500.0,84100.0,85600.0,135700.0,130900.0,68800.0,111400.0,119500.0,60400.0,82400.0,135700.0,177600.0,223000.0,258500.0,167000.0,104900.0,101600.0,85800.0,121700.0,71700.0,92100.0,74300.0,82500.0,78100.0,133500.0,112700.0,88500.0,88500.0,133700.0,33800.0,52400.0,47600.0,62400.0,44600.0,62300.0,49900.0,61800.0,44800.0,29500.0,70000.0,244500.0,220000.0,45200.0,286600.0,218600.0,57400.0,64600.0,167300.0,39500.0,27600.0,16700.0,192400.0,31900.0,70300.0,25100.0,132300.0,107400.0,112600.0,68500.0,91700.0,126400.0,72400.0,67400.0,46100.0,336800.0,148600.0,71500.0,49800.0,49800.0,110000.0,68800.0,78200.0],
        [0.06441732474664304,0.07592680410944491,0.07520767262303545,0.0772849592723677,0.07729428748803809,0.08369054363897388,0.08722961032869522,0.09276626686816761,0.0928071265071059,0.0926012564602298,0.09259145590207653,0.1106313843430622,0.11085572394564648,0.11769914606172255,0.11427428910612249,0.11537181863597647,0.11529827394502026,0.11402299332807367,0.11702909345010301,0.11868714714155165,0.12071002214205764,0.12044662040528292,0.12020544221494439,0.12045896966091453,0.1184561905042781,0.1193893668003695,0.11661237579467283,0.11292329592734536,0.10836314903195668,0.10867996271290659,0.11235577161547555,0.11217493850301613,0.08816218541195102,0.08913998117102669,0.0879520164039672,0.09733985025653391,0.10233584082837836,0.10370034328981194,0.10423369271510308,0.10168148695256563,0.1000271279244772,0.0990584091589674,0.09866941353381395,0.09282653224447623,0.09585809212309515,0.09261304726780174,0.09447676636429687,0.09754621292506971,0.09751848863908744,0.10095778688084285,0.10267648484934454,0.10061191544190992,0.10623969048272705,0.10602195997771259,0.1106675462229492,0.10498241533614036,0.09810641952898036,0.09308415796818166,0.09268321909353841,0.08876783037093448,0.08768198840156645,0.08786451577026204,0.08441659090659978,0.08560859982105946,0.08099826500710484,0.0792138137067806,0.07778904348723653,0.07429077110679555,0.07609067878541444,0.07456198616126887,0.07005266602146401,0.07265753045271323,0.07504948255015137,0.07241716885263869,0.0748448149242451,0.08133288392479508,0.08460611619768753,0.08788407029002453,0.0901422436203747,0.08817959912774394,0.08823123710622925,0.09675315159515806,0.0967354143580137,0.09652357283146974,0.0972254340248625,0.09692090893396789,0.09491064517930749,0.09477187427306544,0.10852479993119914,0.10984652804892489,0.11026815769347886,0.11117653073893162,0.10760005579020371,0.10620083977316311,0.1028484693906359,0.10323979799694051,0.08996576116417829,0.08393248580735932,0.07668621465981629,0.07697963243006606,0.07443491879743529,0.07663646207506389,0.07122609805869977,0.07175310597373928,0.07392890946986809,0.07411264214363451,0.07394450133431492,0.07475546343579058,0.07720922974720003,0.06395329154271966,0.055727607427339454,0.05636874416368321,0.05355887125114047,0.06298003280360535,0.0630136099937857,0.06952854806318273,0.06954640857116459,0.07615783392748232,0.07752104000334196,0.07743842503486496,0.07731032711956953,0.08780700714926098,0.08586784597735331,0.08516550335010156,0.10207151435887427,0.10369922456743752,0.10500337417486469,0.10696684464202635,0.11643034766066587,0.11676723656792452,0.1160466209769425,0.11974092783771492,0.12089087543421977,0.12119811601761152,0.12288878235320246,0.12404095945049956,0.11893761367538958,0.11915131213203209,0.11439729204977611,0.11197025633795105,0.11344279894912945,0.11416070029015798,0.11235466870157215,0.11241849458328182,0.11859319917885512,0.10533309888666333,0.10600775217600236,0.10974333731414307,0.10595082458789104,0.08494916695082916,0.08526551257970143,0.08513456025486223,0.08560265928782443,0.08308928900266506,0.08625287939130852,0.08391513438673938,0.07690991322396731,0.07684763693481275,0.08114028767942251,0.08420864073953542,0.0850008072878087,0.08499722999614645,0.10153161679068555,0.09029155324339365,0.09094692101227798,0.12453011596145941,0.13244790881615248,0.13164922137982124,0.1303949796075627,0.13152746512677962,0.14214646919122134,0.21545833573198928,0.22123217089039732,0.22105940477049696,0.22887043025370293,0.2287743034319295,0.22672266868104612,0.2256213659089477,0.2252469462874198,0.21785229637110312,0.21298075053472934,0.22713960056753596,0.22564921042194036,0.2300720712572095,0.23077574361995715,0.23060144709480668,0.21989501851915402,0.2179883299201872,0.2185922070842535,0.2163683778233405,0.2263153753065462,0.22119895288328437,0.14783153280086098,0.14290837566156595,0.15059640116084516,0.13164218867990926,0.13562647547163584,0.13578149221587346,0.13747294648631056,0.13761490886749048,0.1377553194721631,0.13764592446829693,0.1193097796434351,0.13306812734648638,0.12656549747776638,0.13481651424991445,0.1434115599053037,0.14353543589137094,0.14441701970530818,0.1443521103267933,0.14699056273554637,0.13443774237926762,0.13372770840203227,0.1271160964413127,0.1308267060205698,0.11906909743238131,0.11482264528474366,0.11306271225596325,0.11310785529562581,0.11267669188244955,0.11769464220849088,0.11839799159218925,0.12751932260476448,0.14850041980954895,0.14080299107816685,0.1410615123618069,0.12906653721341457,0.12254215117603083,0.12320941689280163,0.12425427369905002,0.1227426900279964,0.1216105600509703,0.12566595975883327,0.127674569002761,0.1307815574799334,0.12895746584667586,0.13002721326042085,0.12928227641420295,0.135976412106799,0.14142906236836306,0.1455425474400031,0.1406240156263148,0.14228991157927348,0.13478069582770313,0.11816265318145071,0.11312084107993998,0.11305014355516624,0.11212751414401119,0.11121918395508047,0.11156847731469013,0.1092936574079869,0.1267412560729183],
        [77.1500015258789,77.25,76.8499984741211,76.9000015258789,77.69999694824219,78.55000305175781,79.44999694824219,79.55000305175781,79.75,79.69999694824219,81.4000015258789,81.30000305175781,80.55000305175781,80.6500015258789,80.4000015258789,80.44999694824219,80.0,80.75,80.4000015258789,79.94999694824219,80.3499984741211,81.55000305175781,81.55000305175781,82.19999694824219,82.05000305175781,82.19999694824219,82.4000015258789,82.69999694824219,82.6500015258789,82.0999984741211,82.30000305175781,82.44999694824219,82.1500015258789,83.0,82.0999984741211,81.3499984741211,81.80000305175781,82.5,82.19999694824219,82.3499984741211,82.94999694824219,82.9000015258789,82.1500015258789,81.5999984741211,81.25,80.75,81.30000305175781,81.0,81.6500015258789,81.1500015258789,81.05000305175781,81.8499984741211,81.75,82.55000305175781,82.69999694824219,82.55000305175781,82.30000305175781,82.69999694824219,82.94999694824219,83.0,82.8499984741211,82.94999694824219,82.5999984741211,82.25,82.69999694824219,82.55000305175781,82.8499984741211,82.30000305175781,82.3499984741211,82.3499984741211,83.1500015258789,82.75,82.19999694824219,82.69999694824219,81.5999984741211,80.94999694824219,80.30000305175781,80.69999694824219,80.4000015258789,80.6500015258789,81.5,81.5,81.5,81.8499984741211,82.0999984741211,82.0999984741211,82.0999984741211,83.4000015258789,84.19999694824219,84.05000305175781,84.5,84.30000305175781,84.5999984741211,84.9000015258789,84.55000305175781,84.5,84.4000015258789,84.3499984741211,84.25,84.30000305175781,84.94999694824219,85.19999694824219,85.0999984741211,84.80000305175781,84.69999694824219,84.8499984741211,85.30000305175781,84.94999694824219,84.4000015258789,84.55000305175781,84.30000305175781,84.1500015258789,83.30000305175781,83.5999984741211,84.3499984741211,84.0,84.75,85.1500015258789,85.19999694824219,85.1500015258789,84.1500015258789,83.5999984741211,83.69999694824219,82.25,82.5999984741211,82.9000015258789,82.25,80.8499984741211,80.9000015258789,80.9000015258789,81.5,80.94999694824219,81.0,81.75,81.0999984741211,81.19999694824219,80.80000305175781,81.05000305175781,81.0,81.30000305175781,81.44999694824219,80.6500015258789,80.9000015258789,81.69999694824219,82.1500015258789,82.5999984741211,83.3499984741211,83.5,83.75,84.0999984741211,84.1500015258789,83.75,83.4000015258789,84.0999984741211,84.6500015258789,85.0,85.1500015258789,86.30000305175781,87.1500015258789,87.94999694824219,88.30000305175781,87.19999694824219,87.44999694824219,87.5,85.55000305175781,84.55000305175781,84.6500015258789,85.25,84.9000015258789,83.5999984741211,79.6500015258789,80.6500015258789,80.5999984741211,79.05000305175781,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,79.69999694824219,81.5,81.0,82.0,82.4000015258789,82.3499984741211,82.3499984741211,81.75,81.1500015258789,80.75,82.1500015258789,81.80000305175781,82.75,82.8499984741211,84.1500015258789,84.94999694824219,84.30000305175781,84.25,83.9000015258789,83.8499984741211,83.80000305175781,83.8499984741211,83.55000305175781,82.0999984741211,82.19999694824219,83.4000015258789,82.25,82.0999984741211,82.8499984741211,82.25,81.5,81.5,81.5,81.5,82.19999694824219,82.5,82.8499984741211,82.5,82.5,82.30000305175781,81.4000015258789,81.5999984741211,82.6500015258789,80.75,79.94999694824219,79.55000305175781,79.30000305175781,79.05000305175781,79.19999694824219,80.0,80.0,79.4000015258789,78.55000305175781,78.94999694824219,79.5,80.0,80.44999694824219,80.6500015258789,81.5999984741211,82.5,81.6500015258789,81.75,81.19999694824219,80.94999694824219,82.05000305175781,81.75,81.3499984741211,81.55000305175781,81.8499984741211,82.19999694824219,81.8499984741211,80.4000015258789,80.75]
          ] )

    data_a_month = np.array(
        [
         [82.15000153 ,81.80000305 ,82.75       ,82.84999847 ,84.15000153,84.94999695 ,84.30000305 ,84.25       ,83.90000153 ,83.84999847 ,83.80000305,83.84999847 ,83.55000305 ,82.09999847 ,82.19999695 ,83.40000153 ,82.25,82.09999847 ,82.84999847 ,82.25       ,81.5        ,81.5        ,81.5,81.5        ,82.19999695 ,82.5        ,82.84999847 ,82.5        ,82.5,82.30000305 ,81.40000153 ,81.59999847 ,82.65000153 ,80.75       ,79.94999695,79.55000305 ,79.30000305 ,79.05000305 ,79.19999695 ,80.         ,80.,79.40000153 ,78.55000305 ,78.94999695 ,79.5        ,80.         ,80.44999695,80.65000153 ,81.59999847 ,82.5        ,81.65000153 ,81.75       ,81.19999695,80.94999695 ,82.05000305 ,81.75       ,81.34999847 ,81.55000305 ,81.84999847,82.19999695 ,81.84999847 ,80.40000153 ,80.75      ],
         [82400, 135700, 177600, 223000, 258500, 167000, 104900, 101600,           85800, 121700,  71700,  92100,  74300,  82500,  78100, 133500, 112700,           88500,  88500, 133700,  33800,  52400,  47600,  62400,  44600,  62300,           49900,  61800,  44800,  29500,  70000, 244500, 220000,  45200, 286600,          218600,  57400,  64600, 167300,  39500,  27600,  16700, 192400,  31900,           70300,  25100, 132300, 107400, 112600,  68500,  91700, 126400,  72400,           67400,  46100, 336800, 148600,  71500,  49800,  49800, 110000,  68800,           78200],
         [0.21636838, 0.22631538, 0.22119895, 0.14783153, 0.14290838, 0.1505964 , 0.13164219, 0.13562648, 0.13578149, 0.13747295, 0.13761491, 0.13775532, 0.13764592, 0.11930978, 0.13306813, 0.1265655 , 0.13481651, 0.14341156, 0.14353544, 0.14441702, 0.14435211, 0.14699056, 0.13443774, 0.13372771, 0.1271161 , 0.13082671, 0.1190691 , 0.11482265, 0.11306271, 0.11310786, 0.11267669, 0.11769464, 0.11839799, 0.12751932, 0.14850042, 0.14080299, 0.14106151, 0.12906654, 0.12254215, 0.12320942, 0.12425427, 0.12274269, 0.12161056, 0.12566596, 0.12767457, 0.13078156, 0.12895747, 0.13002721, 0.12928228, 0.13597641, 0.14142906, 0.14554255, 0.14062402, 0.14228991, 0.1347807 , 0.11816265, 0.11312084, 0.11305014, 0.11212751, 0.11121918, 0.11156848, 0.10929366, 0.12674126]
        ]
    )

    # data_a_month = np.array(
    #     [
    #      [81.3499984741211, 82.5999984741211, 82.5, 83.30000305175781, 83.44999694824219, 82.55000305175781, 82.6500015258789, 82.5, 83.0, 81.75, 81.94999694824219, 81.94999694824219, 80.5999984741211, 81.5, 81.25, 81.1500015258789, 80.9000015258789, 80.69999694824219, 80.4000015258789, 80.05000305175781, 81.44999694824219],
    #      [102400.0, 120700.0, 138000.0, 126300.0, 72400.0, 68600.0, 110700.0, 89500.0, 83700.0, 163200.0, 64700.0, 113300.0, 53800.0, 135800.0, 167600.0, 96400.0, 151700.0, 132000.0, 149200.0, 65600.0, 54000.0],
    #      [0.11746395710891079, 0.11912344199547836, 0.127298691892572, 0.12659927136235996, 0.12854404661582505, 0.12787718082248525, 0.13470584969830549, 0.1293602252558059, 0.12419032853411432, 0.12008584478857405, 0.13160393287345312, 0.12959973188200063, 0.12901223172892728, 0.13294167419811648, 0.13845024378667828, 0.13783148796494069, 0.13758782819292179, 0.13718159604907482, 0.1362273733863873, 0.1360776012977379, 0.12281450118727147]
    #     ]
    # )

    # print(data_to_use[0][-63:])
    # print(data_to_use[1][-63:])
    # print(data_to_use[2][-63:])
    lstm_filepath = 'D:/Alia/Documents/asset allocation/output/test/' # 存lstm預測績效圖
    # filename = 'ecm_lstm_0906.png'
    # mse, predict_price = lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'lstm_0919.png',lstm_filepath)
    # mse, predict_price = atten_lstm_corr(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'atten_lstm_0922.png',lstm_filepath)
    # mse, predict_price = atten_lstm_pred1m(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'atten_lstm_0923.png',lstm_filepath)
    # mse, predict_price = ecm_lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'ecm_lstm_0911.png',lstm_filepath)
    # mse, predict_price = atten_ecm_lstm(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'atten_ecm_lstm_0911.png',lstm_filepath)
    atten_lstm_out1m(batchsz,units,epochs,window_size_x,window_size_y,data_to_use,data_a_month,'atten_lstm_0923.png',lstm_filepath)
    # print(predict_price)

    end = time.time()
    print(end-start)

# %%
