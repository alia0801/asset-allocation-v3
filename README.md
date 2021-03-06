# asset allocation

### 資料庫
```
myetf
|-- close
|-- detail(有補滿管理費、總費用率資料)
|-- etf_close
|-- etf_volume
|-- my_etf
|-- record
|-- volume
|-- 各長年化值
```

### 程式
* 資料庫建立
    * `DB_create.py`
* 演算法相關
    * 演算法
        * 目前使用
            * `bl_weight.py` 有目前演算法會call的函式，可用bl模型將觀點矩陣轉成權重
            * `choose.py` 產出組合的主程式
            * `clustering_corr.py` 將所有標的依相關性分為特定群數的函式
            * `clustering_type.py` 將所有標的依性質分群的函式
            * `generate_input_data.py` 進行分群、產生各群的平均收盤價、波動度、交易量
            * `price2matrix.py` 將預測出來的價格轉為觀點矩陣的函式
            * `new_rnn.py` 包含目前運行lstm與ecm-lstm的函式
            * `txo.R` 相關性分群
        * 其他
            * `bl_model.py` 
            * `ecm.py` 實作ECM
            * `lstm.py` 以close預測close
            * `lstm_3input.py` 以波動度、close、volume預測close
            * `main.py` 舊的產出組合主程式
            * `new_lstm.py` 實作ecm-lstm，以波動度、close、volume預測close
            * `rnn.py` 舊的lstm函式
            
    * baseline
        * 目前使用
            * `choose_avg.py` 產出平均權重組合的主程式
            * `choose_timeseries.py` 以傳統時序分析方式產出組合的主程式:以ARIMA或Holt-winters預測價格(call `time_series_predict`的函式)、轉為觀點矩陣、套入bl模型計算權重並輸出
            * `paper_baseline.py` 分群後以mvp或mvtp產生組合的主程式
            * `time_series_predict.py` 以傳統時序分析方式預測價格的函式
        * 其他
            * `arima.py`
            * `arima_auto.py` 目前ARIMA使用的方式
            * `holtwinter.py` 目前holt-winters使用的方式
            * `mvp.py` 目前mvp與mvtp的使用方式
* 實驗相關
    * 目前使用
        * `evaluation_rebalance_new.py` 有模擬金流投入及計算績效的程式
        * `experience_new.py` 計算績效的主程式
    * 舊的
        * `evaluation_rebalance.py` 
        * `experience_old.py` 
        * `experience.py` 

* 實驗方式
    1. 產生組合的csv檔
        * 指令:`python <py檔(如choose.py)> <起始年> <起始月> <跑哪個部分> <跑幾個月>`
        <跑哪個部分>: 1=一次分群+動態調整 2=讀前面存下的csv繼續動態調整 3=給定標的只調整比例
        例如: `python choose.py 2015 1 3 2` 表示從2015年1月開始，使用給定的標的，動態調整比例2個月
    2. 產生圖表
        * 使用`evaluation_rebalance_new.py`
        

