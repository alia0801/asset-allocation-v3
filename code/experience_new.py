# %%
# import choose
import evaluation_rebalance_new
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import math
import statistics
import numpy as np

# %%
# 
class Ans_comb:
    def _init_(self,input_money,sum_money,the_names,the_weight,train_reward,train_ann_reward,train_risk,train_mdd,success_ratio,success,stat_list,success_stat,test_reward,test_ann_reward,test_risk,test_mdd):
        self.the_names = the_names
        self.the_weight = the_weight
        self.train_reward = train_reward
        self.train_ann_reward = train_ann_reward
        self.train_risk = train_risk
        self.train_mdd = train_mdd
        self.success_ratio = success_ratio
        self.success = success
        self.stat_list = stat_list
        self.success_stat = success_stat
        self.test_reward = test_reward
        self.test_ann_reward = test_ann_reward
        self.test_risk = test_risk
        self.test_mdd = test_mdd
        self.test_sum_money = sum_money
        self.input_money = input_money

    def print_test_evalu(self):
        print(self.the_names)    
        print(self.the_weight) 
        print('test_ann_reward',self.test_ann_reward) 
        print('test_reward',self.test_reward) 
        print('test_risk',self.test_risk)
        print('test_mdd',self.test_mdd)
        # print('success_ratio',self.success_ratio)   
        # print('stat_list',self.stat_list)
        # print('success_stat',self.success_stat)   

# %%
colors = ['green','black','orange','purple','pink','yellow','blue','gray','yellowgreen']
tw50 = ['0050.TW','1.0']
twii = ['006204.TW','1.0']
spy = ['SPY','1.0']
black_swan = ['00679B.TW 00635U.TW 00682U.TW 00706L.TW','0.2491 0.3086 0.217 0.2253' ] # black swan
grow_yuanta = ['00646.TW 0050.TW 006206.TW 00661.TW 00660.TW 00697B.TW 00720B.TW 00635U.TW 00682U.TW', '0.1791 0.2301 0.1082 0.0747 0.0458 0.1150 0.0271 0.0782 0.1418'] # 成長
green_horn_comb = ['VTI VGK VPL VWO BWX IEI','0.15 0.15 0.15 0.15 0.2 0.2']
mr_market_classic = ['ITOT VEU VNQ AGG','0.36 0.18 0.06 0.4']

# evaluation_value_list = ['reward','annual_reward','v1_std','mdd','success_ratio','success']
# %%
# train 約y-1/month/1 ~ y/month/1(共252日)
def train_evaluation(y,month,ans,mode,first_input_total,input_month_total):
    
    # y_train = y-nnnn
    nnnn = 1
    y_train = y
    comb_list = []
    for a in range(len(ans)):
        choose1 = ans[a][0]
        weight1 = ans[a][1]
    
        choose_name = choose1.split(' ')
        weight = weight1.split(' ')
        for i in range(len(weight)):
            weight[i] = float(weight[i])
    
        evaluation_value = evaluation_rebalance_new.train_choose(y_train,month,choose_name,weight,first_input_total,input_month_total,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        if evaluation_value is None:
            continue
        
        tmp = Ans_comb()
        tmp.the_names = choose_name
        tmp.the_weight = weight
        tmp.train_reward = evaluation_value[0]
        tmp.train_ann_reward = evaluation_value[1]
        tmp.train_risk = evaluation_value[2]
        tmp.train_mdd = evaluation_value[3]
    
        comb_list.append(tmp)
        
        # print(choose_name)
        # print(weight)
        # # print('answers:')
        # for i in range(1,len(evaluation_value)):
        #     print(evaluation_value_list[i],evaluation_value[i])
        # print()
        
    
    
    # train evaluation
    # multi-criteria
    if len(comb_list)>0:
        gotest_candidate = [comb_list[0]]
    else:
        gotest_candidate = []
    
    for a in range(1,len(comb_list)):
        go = True
        tmp_1 = comb_list[a] # 未入
        bad = []
        for b in range(len(gotest_candidate)):
            tmp_2 = gotest_candidate[b] #已入
            if( (tmp_1.train_ann_reward >= tmp_2.train_ann_reward) 
                and (tmp_1.train_risk <= tmp_2.train_risk) 
                and (tmp_1.train_mdd <= tmp_2.train_mdd) 
                # and (tmp_1.success_ratio >= tmp_2.success_ratio) 
                ):
                # 未入 優
                go = True
                bad.append(b)
            if( (tmp_1.train_ann_reward <= tmp_2.train_ann_reward) 
                and (tmp_1.train_risk >= tmp_2.train_risk) 
                and (tmp_1.train_mdd >= tmp_2.train_mdd) 
                # and (tmp_1.success_ratio <= tmp_2.success_ratio) 
                ):
                # 已入 優
                go = False
        if len(bad)>0:
            bad.reverse()
            for b in bad:
                del gotest_candidate[b]
        if go :
            gotest_candidate.append(tmp_1)    
    print('gotest_candidate: ')
    for b in range(len(gotest_candidate)):
        print(gotest_candidate[b].the_names) 

    return comb_list,gotest_candidate  

# %%
# test y/month/1 ~ 約y/month+1/1(共21日)-->先改看1年績效
def test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month,input_month_total):
    nnnn = 1
    y_test = y
    for a in range(len(gotest_candidate)):

        tmp = gotest_candidate[a]
        choose_name = tmp.the_names
        weight = tmp.the_weight

        if nochange ==0:
            evaluation_value = evaluation_rebalance_new.test_choose(y_test,month,choose_name,weight,first_input_total,true_first_in,input_month_total,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        else:
            evaluation_value = evaluation_rebalance_new.test_choose_nochange(y_test,month,nnn_month,choose_name,weight,first_input_total,input_month_total,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
                                                        # test_choose_nochange(y,month,nnn_month,choose,weight,first_input_total,mode=3)
        # success_ratio,success,stat_list,stat = evaluation_rebalance.slidwindow(y_test,nnnn,choose_name,weight,expect_reward,mode)
        # evaluation_value.append(success_ratio)
        # evaluation_value.append(success)
        # evaluation_value.append(stat_list)
        # evaluation_value.append(stat)
        
        # if evaluation_value is None:
            # evaluation_value = [-100,-100,100,100,[]]
        tmp.test_reward = evaluation_value[0]
        tmp.test_ann_reward = evaluation_value[1]
        tmp.test_risk = evaluation_value[2]
        tmp.test_mdd = evaluation_value[3]
        tmp.test_sum_money = evaluation_value[4]
        tmp.input_money = evaluation_value[5]
        # tmp.success_ratio = evaluation_value[6]
        # tmp.success = evaluation_value[7]
        # tmp.success_stat = evaluation_value[8]
        # tmp.stat_list = evaluation_value[9]


        # print(choose_name)
        # print(weight)
        # # print('answers:')
        # for i in range(1,len(evaluation_value)):
        #     print(evaluation_value_list[i],evaluation_value[i])
        # print()


    # test evaluation
    # multi-criteria
    if len(gotest_candidate)>0:
        final_ans_candidate = [gotest_candidate[0]]
    else:
        final_ans_candidate = []

    for a in range(1,len(gotest_candidate)):
        go = True
        tmp_1 = gotest_candidate[a]
        if tmp_1.test_ann_reward==-100:
            continue
        bad = []
        for b in range(len(final_ans_candidate)):
            tmp_2 = final_ans_candidate[b]
            if tmp_2.test_ann_reward==-100:
                continue
            # if( (tmp_1.test_ann_reward >= tmp_2.test_ann_reward) 
            if( (tmp_1.test_reward >= tmp_2.test_reward) 
                and (tmp_1.test_risk <= tmp_2.test_risk) 
                and (tmp_1.test_mdd <= tmp_2.test_mdd) 
                # and (tmp_1.success_ratio >= tmp_2.success_ratio) 
                ):
                # final_ans_candidate[b] = tmp_1
                # go = False
                # break
                go = True
                bad.append(b)
            # elif( (tmp_1.test_ann_reward <= tmp_2.test_ann_reward) 
            elif( (tmp_1.test_reward <= tmp_2.test_reward) 
                and (tmp_1.test_risk >= tmp_2.test_risk) 
                and (tmp_1.test_mdd >= tmp_2.test_mdd) 
                # and (tmp_1.success_ratio <= tmp_2.success_ratio) 
                ):
                # final_ans_candidate[b] = tmp_2
                go = False
                # break
        if len(bad)>0:
            bad.reverse()
            for b in bad:
                del final_ans_candidate[b]
        if go :
            final_ans_candidate.append(tmp_1)    

    print('final_ans_candidate: ')
    for b in range(len(final_ans_candidate)):
        print(final_ans_candidate[b].the_names) 
        # plt.plot(final_ans_candidate[b].test_sum_money, color='b')
        # plt.show()



    # use = 0
    if len(final_ans_candidate)>1:
        if use == 0:
            # use test_ann_reward
            final_ann_reward = 0
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                # if tmp.test_ann_reward == final_ann_reward:
                if tmp.test_reward == final_ann_reward:
                    # print(tmp.test_ann_reward,final_ann_reward)
                    final_comb.append(tmp)
                # elif tmp.test_ann_reward > final_ann_reward: 
                elif tmp.test_reward > final_ann_reward: 
                    final_comb = [tmp]
                    final_ann_reward = tmp.test_reward
                    # final_ann_reward = tmp.test_ann_reward
            # print(final_ann_reward)
            if len(final_comb)==0:
                final_ann_reward = -100
                # final_comb=[]
                for a in range(len(final_ans_candidate)):
                    tmp = final_ans_candidate[a]
                    # if tmp.test_ann_reward == final_ann_reward:
                    if tmp.test_reward == final_ann_reward:
                        # print(tmp.test_ann_reward,final_ann_reward)
                        final_comb.append(tmp)
                    # elif tmp.test_ann_reward > final_ann_reward: 
                    elif tmp.test_reward > final_ann_reward: 
                        final_comb = [tmp]
                        final_ann_reward = tmp.test_reward
                        # final_ann_reward = tmp.test_ann_reward
        elif use == 1:
            # use test_risk
            final_risk = 1000
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                # if tmp.test_risk == final_risk and tmp.test_ann_reward >= expect_reward:
                if tmp.test_risk == final_risk :
                    final_comb.append(tmp)
                # elif tmp.test_risk < final_risk and tmp.test_ann_reward >= expect_reward:
                elif tmp.test_risk < final_risk :
                    final_comb = [tmp]
                    final_risk = tmp.test_risk
            if len(final_comb)==0:
                final_risk = 1000
                final_comb=[]
                for a in range(len(final_ans_candidate)):
                    tmp = final_ans_candidate[a]
                    if tmp.test_risk == final_risk:
                        final_comb.append(tmp)
                    elif tmp.test_risk < final_risk:
                        final_comb = [tmp]
                        final_risk = tmp.test_risk
        elif use == 2:
            # use test_mdd
            final_mdd = 1000
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                if tmp.test_mdd == final_mdd:
                    final_comb.append(tmp)
                elif tmp.test_mdd < final_mdd: 
                    final_comb = [tmp]   
                    final_mdd = tmp.test_mdd
        # elif use == 3:
        #     # use success_ratio
        #     final_success_ratio = 0
        #     final_comb=[]
        #     for a in range(len(final_ans_candidate)):
        #         tmp = final_ans_candidate[a]
        #         if tmp.success_ratio == final_success_ratio:
        #             final_comb.append(tmp)
        #         elif tmp.success_ratio > final_success_ratio:    
        #             final_comb = [tmp]  
        #             final_success_ratio = tmp.success_ratio
        elif use == 4:
            final_risk = 1000
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                if tmp.test_risk == final_risk:
                    final_comb.append(tmp)
                elif tmp.test_risk < final_risk:
                    final_comb = [tmp]
                    final_risk = tmp.test_risk

    else:
        if len(final_ans_candidate)>0:
            final_comb=[final_ans_candidate[0]]
        else:
            final_comb=[]

    print('final_comb:')
    for b in range(len(final_comb)):
        # print(final_comb[b].the_names) 
        # print(final_comb[b].test_ann_reward) 
        final_comb[b].print_test_evalu()

    return gotest_candidate,final_ans_candidate,final_comb

# %%

def calculate_values(final_input_money,final_sum_money):
    reward = (final_sum_money[-1]-final_input_money[-1])/final_input_money[-1]
    
    # print(final_sum_money,final_input_money)
    m_rews = []
    count_month = 1
    input = final_input_money[0]
    for i in range(len(final_sum_money)):
        if final_input_money[i]>input:
            # print(final_sum_money[i-1],input)
            rrr = (final_sum_money[i-1]-input)/input
            input = final_input_money[i]
            count_month+=1
            m_rews.append(rrr)
    rrr = (final_sum_money[-1]-input)/input 
    m_rews.append(rrr) 
    nnnn = count_month/12
    # print(m_rews)
    
    std_dev_m = statistics.stdev(m_rews)
    v1_std = std_dev_m * math.pow( 12, 0.5 )
    # print('std_dev',v1_std)

    annual_reward = math.pow( (1+reward), 1/nnnn )-1

    # day_return = [] #len = len(final_sum_money)-1
    # tmp = 0
    # for i in range(1,len(final_sum_money)):
    #     if final_sum_money[i-1]!=0:
    #         tmp = (final_sum_money[i]-final_sum_money[i-1])/final_sum_money[i-1]
    #         day_return.append( tmp )
    #     else:
    #         day_return.append( tmp )
    # print(day_return)
    
    max_price = [] #len = len(final_sum_money)-1
    for i in range(len(final_sum_money)):
        # tmp = day_return[0:i+1].max()
        tmp = np.max(np.array(final_sum_money[0:i+1]))
        max_price.append(tmp)
    # print(max_price)

    dd = []
    for i in range(len(max_price)):
        tmp = 1-(final_sum_money[i]/max_price[i])
        dd.append(tmp)
    # print(dd)
    # mdd = dd.max()
    mdd = np.max(np.array(dd))

    values = [reward,annual_reward,v1_std,mdd]
    print(values)

    return values

# %%
def run_once_money_sim_market(ans,ans_new,market,first_input_total,y,month,use=1,mode=4):
    nnn_month = len(ans_new)+1
    true_first_in = first_input_total
    # print(nnn_month)
    # 投組為大盤
    if market == 'tw':
        ans_market = [twii] # 大盤
    elif market == 'us':
        ans_market = [spy] # 大盤
    else:
        ans_market = [ans]
    # print(ans_market)
    nochange = 1
    comb_list,gotest_candidate = train_evaluation(y,month,ans_market,mode,first_input_total,input_month_total)
    gotest_candidate,final_ans_candidate,final_comb_market = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month,input_month_total)
                                                            # test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
    return final_comb_market

# %%
def run_once_money_sim(ans,ans_new,market,first_input_total,y,month,use=1,mode=4):
    nnn_month = len(ans_new)+1
    true_first_in = first_input_total
    # first_input_total = 150000

    # 投組第一次選出
    nochange = 0
    comb_list,gotest_candidate = train_evaluation(y,month,ans,mode,first_input_total,input_month_total)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month,input_month_total)
    input_money_record = [final_comb[0].input_money]
    sum_money_record = [final_comb[0].test_sum_money]
    
    # 動態平衡
    for i in range(len(ans_new)):
        true_first_in += input_month_total
        first_input_total = final_comb[0].test_sum_money[-1]
        # first_input_total = final_comb[0].test_sum_money[-1]+10000
        if month!=12:
            month +=1
        else:
            y+=1
            month=1

        comb_list,gotest_candidate = train_evaluation(y,month,[ans_new[i]],mode,first_input_total,input_month_total)
        gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month,input_month_total)
        input_money_record.append(final_comb[0].input_money)
        sum_money_record.append(final_comb[0].test_sum_money)
        # print()
    final_input_money = [j for sub in input_money_record for j in sub]
    final_sum_money = [j for sub in sum_money_record for j in sub]
    
    # # 投組為大盤
    # if market == 'tw':
    #     ans_market = [twii] # 大盤
    # elif market == 'us':
    #     ans_market = [spy] # 大盤
    # nochange = 1
    # comb_list,gotest_candidate = train_evaluation(y,month,ans_market,mode,first_input_total)
    # gotest_candidate,final_ans_candidate,final_comb_market = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
    # # final_input_money_market = final_comb[0].input_money
    # # final_sum_money_market = final_comb[0].test_sum_money

    return final_input_money,final_sum_money

# %%

def calculate_combs(y,month,market,first_input_total,ans,ans_new):
    compared_combs = []
    
    final_input_money,final_sum_money = run_once_money_sim(ans,ans_new,market,first_input_total,y,month)
    final_comb_market = run_once_money_sim_market(ans,ans_new,market,first_input_total,y,month,use=1,mode=4)
    compared_combs.append(final_comb_market)

    if market=='us':
        labels = ['Market','Green Horn','Mr.Market']
        exist_comb = [green_horn_comb,mr_market_classic]
        # labels = ['Market','Green Horn']
        # exist_comb = [green_horn_comb]
        for comb in exist_comb:
            final_comb_exist = run_once_money_sim_market(comb,ans_new,'other',first_input_total,y,month,use=1,mode=4)
            compared_combs.append(final_comb_exist)

    elif market=='tw':
        labels = ['Market','Black Swan','grow-Yuanta']
        exist_comb = [black_swan,grow_yuanta]
        for comb in exist_comb:
            final_comb_exist = run_once_money_sim_market(comb,ans_new,'other',first_input_total,y,month,use=1,mode=4)
            compared_combs.append(final_comb_exist)

    return final_input_money,final_sum_money,compared_combs,labels


# %%

def plot_money_sim(filepath,final_input_money,record_comb_moneysim,compared_combs,labels):
    
    fig_y_min = 50000
    fig_y_max = 300000

    plt.plot(final_input_money, color='red',label='input')

    for i in range(len(compared_combs)):
        comb = compared_combs[i]
        plt.plot(comb[0].test_sum_money, color=colors[i],label=labels[i])

    for i in range(len(record_comb_moneysim)):
        final_sum_money = record_comb_moneysim[i]
        plt.plot(final_sum_money, color=colors[i+len(compared_combs)],label=labels[i+len(compared_combs)])

    plt.ylim([fig_y_min,fig_y_max])
    plt.legend(loc='upper left')
    plt.savefig(filepath+'money sim-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
    # plt.show()
    plt.clf()
    plt.close()


# %%
def cal_ann_reward_list(input_money,sum_money):

    reward_list = [0]
    input = input_money[0]
    count_month = 1
    for i in range(len(sum_money)):
        if input_money[i]>input:
            r = (sum_money[i-1]-input)/input
            input = input_money[i]
            nnnn = count_month/12
            ann_r = math.pow( (1+r), 1/nnnn )-1
            reward_list.append(ann_r)
            count_month += 1
    # print(len(sum_money),i)
    r = (sum_money[i-1]-input)/input
    input = input_money[i]
    nnnn = count_month/12
    ann_r = math.pow( (1+r), 1/nnnn )-1
    reward_list.append(ann_r)

    return reward_list

# %%
def plot_ann_reward(filepath,final_input_money,record_comb_moneysim,compared_combs,labels):

    fig_y_min = -1
    fig_y_max = 1.5

    for i in range(len(compared_combs)):
        comb = compared_combs[i]
        reward_list = cal_ann_reward_list(final_input_money,comb[0].test_sum_money)
        plt.plot(reward_list, color=colors[i],label=labels[i])

    for i in range(len(record_comb_moneysim)):
        final_sum_money = record_comb_moneysim[i]
        reward_list = cal_ann_reward_list(final_input_money,final_sum_money)
        plt.plot(reward_list, color=colors[i+len(compared_combs)],label=labels[i+len(compared_combs)])

    plt.ylim([fig_y_min,fig_y_max])
    plt.legend()
    plt.savefig(filepath+'ann reward-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
    # plt.show()
    plt.clf()
    plt.close()

# %%

def compare_plot_3D(filepath,comb_values,legends,compared_combs):
    
    rewards = []
    risks = []
    mdds = []
    for comb in compared_combs:
        rewards.append(comb[0].test_reward)
        risks.append(comb[0].test_risk)
        mdds.append(comb[0].test_mdd)
    for value in comb_values:
        rewards.append(value[0])
        risks.append(value[2])
        mdds.append(value[3])

    print(rewards,risks,mdds)

    p=[]
    ax = plt.subplot(projection='3d')
    for i in range(len(rewards)):
        ttt_fig = ax.scatter([risks[i]],[mdds[i]],[rewards[i]],  marker='o', s=40 ,c=colors[i])
        p.append(ttt_fig)
    ax.set_zlabel('reward') 
    ax.set_ylabel('mdd')
    ax.set_xlabel('risk')
    ax.legend(p,legends)
    plt.savefig(filepath+'3Dfig-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
    # plt.show()
    plt.clf()
    plt.close()

    return rewards,risks,mdds

# %%
def plot_3D(filepath,final_input_money,record_comb_moneysim,compared_combs,labels):
    
    comb_values = []
    for final_sum_money in record_comb_moneysim:
        value = calculate_values(final_input_money,final_sum_money)
        comb_values.append(value)
    legends = labels
    rewards,risks,mdds = compare_plot_3D(filepath,comb_values,legends,compared_combs)

    return legends,rewards,risks,mdds

# %%

def save_performance_metrixs(filepath,legends,rewards,risks,mdds):
    # print('performance metrixs:')
    # print('comb','reward','risk','mdd')
    df_list = []
    for i in range(len(legends)):
        tmp = [legends[i],rewards[i],risks[i],mdds[i]]
        df_list.append(tmp)
        # print(legends[i],rewards[i],risks[i],mdds[i])
        df = pd.DataFrame(df_list,columns=['comb','reward','risk','mdd'])
    df.to_csv(filepath+'performance-'+market+'-'+str(y)+'-'+str(month)+'.csv',index=False)
    # print(df)


# %%

if __name__ == '__main__':
    
    # filepath = 'D:/Alia/Documents/asset allocation/plt/experiance/'
    (y,month,mode,use) = (2018,6,4,1)
    first_input_total = 150000
    input_month_total = 1
    market = 'us'
    # market = 'tw'
##################### 批次處理 ######################
 
    paths = ['us-ew','us-hw','us-lstm','us-mvp','us-mvtp']
    # paths = ['us-maxreward','us-maxSharpe']
    filepath_test = 'D:/Alia/Documents/asset allocation/output/performance/prove del etf/test/'
    filepath_train = 'D:/Alia/Documents/asset allocation/output/performance/prove del etf/train/'
    ans_path = 'D:/Alia/Documents/asset allocation/output/answer/scale+cut/'
    allFileName = os.listdir(ans_path+paths[0]+'/')

    # test
    for f in allFileName:
        print(f)
        record_comb_moneysim = []
        for p in paths:
            print(p)
            # ans_path = 'D:/Alia/Documents/asset allocation/output/answer/'+p+'/' # 預測好的答案放在哪裡
            ans_df = pd.read_csv(ans_path+p+'/'+f)
            y = ans_df['year'][0]
            month = ans_df['month'][0]
            ans = [[ans_df['names'][0],str(ans_df['weights'][0])]]
            ans_new = []
            for i in range(1,len(ans_df)):
                tmp = [ans_df['names'][i],str(ans_df['weights'][i])]
                ans_new.append(tmp)

            print(ans)
            print(ans_new)
            # 金流模擬計算
            final_input_money,final_sum_money,compared_combs,labels = calculate_combs(y,month,market,first_input_total,ans,ans_new)
            # record_comb_performance.append([p,final_input_money,final_sum_money])
            record_comb_moneysim.append(final_sum_money)
        
        for i in range(len(record_comb_moneysim)):
            final_sum_money = record_comb_moneysim[i]
            record_comb_moneysim[i] = final_sum_money[:len(final_input_money)]
        labels = labels + ['Equal Weight','Holt-Winters','LSTM','MVP','MVTP']
        # labels = labels + ['Max Reward','Max Sharpe']
        
        plot_money_sim(filepath_test,final_input_money,record_comb_moneysim,compared_combs,labels)
        legends,rewards,risks,mdds = plot_3D(filepath_test,final_input_money,record_comb_moneysim,compared_combs,labels) # 畫3D圖
        save_performance_metrixs(filepath_test,legends,rewards,risks,mdds) # 存performance metrix
        plot_ann_reward(filepath_test,final_input_money,record_comb_moneysim,compared_combs,labels) # 畫年化報酬率折線圖
        
        # break

    # train
    for f in allFileName:
        print(f)
        record_comb_moneysim = []
        for p in paths:
            print(p)
            # ans_path = 'D:/Alia/Documents/asset allocation/output/answer/'+p+'/' # 預測好的答案放在哪裡
            ans_df = pd.read_csv(ans_path+p+'/'+f)
            if int(ans_df['month'][0])==1:
                y = int(ans_df['year'][0])-1
                month = 12
            else:
                y = ans_df['year'][0]
                month = ans_df['month'][0]-1
            ans = [[ans_df['names'][0],str(ans_df['weights'][0])]]
            ans_new = []
            for i in range(1,len(ans_df)):
                tmp = [ans_df['names'][i],str(ans_df['weights'][i])]
                ans_new.append(tmp)

            # print(ans)
            # print(ans_new)
            # 金流模擬計算
            final_input_money,final_sum_money,compared_combs,labels = calculate_combs(y,month,market,first_input_total,ans,ans_new)
            # record_comb_performance.append([p,final_input_money,final_sum_money])
            record_comb_moneysim.append(final_sum_money)
        
        for i in range(len(record_comb_moneysim)):
            final_sum_money = record_comb_moneysim[i]
            record_comb_moneysim[i] = final_sum_money[:len(final_input_money)]
        labels = labels + ['Equal Weight','Holt-Winters','LSTM','MVP','MVTP']
        # labels = labels + ['Max Reward','Max Sharpe']

        plot_money_sim(filepath_train,final_input_money,record_comb_moneysim,compared_combs,labels)
        legends,rewards,risks,mdds = plot_3D(filepath_train,final_input_money,record_comb_moneysim,compared_combs,labels) # 畫3D圖
        save_performance_metrixs(filepath_train,legends,rewards,risks,mdds) # 存performance metrix
        plot_ann_reward(filepath_train,final_input_money,record_comb_moneysim,compared_combs,labels) # 畫年化報酬率折線圖
        
        # break    

# %%
