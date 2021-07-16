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
def train_evaluation(y,month,ans,mode,first_input_total):
    
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
    
        evaluation_value = evaluation_rebalance_new.train_choose(y_train,month,choose_name,weight,first_input_total,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
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
def test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month):
    nnnn = 1
    y_test = y
    for a in range(len(gotest_candidate)):

        tmp = gotest_candidate[a]
        choose_name = tmp.the_names
        weight = tmp.the_weight

        if nochange ==0:
            evaluation_value = evaluation_rebalance_new.test_choose(y_test,month,choose_name,weight,first_input_total,true_first_in,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        else:
            evaluation_value = evaluation_rebalance_new.test_choose_nochange(y_test,month,nnn_month,choose_name,weight,first_input_total,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
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
    comb_list,gotest_candidate = train_evaluation(y,month,ans_market,mode,first_input_total)
    gotest_candidate,final_ans_candidate,final_comb_market = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
                                                            # test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
    return final_comb_market

# %%
def run_once_money_sim(ans,ans_new,market,first_input_total,y,month,use=1,mode=4):
    nnn_month = len(ans_new)+1
    true_first_in = first_input_total
    # first_input_total = 150000

    # 投組第一次選出
    nochange = 0
    comb_list,gotest_candidate = train_evaluation(y,month,ans,mode,first_input_total)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
    input_money_record = [final_comb[0].input_money]
    sum_money_record = [final_comb[0].test_sum_money]
    
    # 動態平衡
    for i in range(len(ans_new)):
        true_first_in += 10000
        first_input_total = final_comb[0].test_sum_money[-1]+10000
        if month!=12:
            month +=1
        else:
            y+=1
            month=1

        comb_list,gotest_candidate = train_evaluation(y,month,[ans_new[i]],mode,first_input_total)
        gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,true_first_in,nochange,nnn_month)
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
        labels = ['comb','market','green horn','mr.market']
        exist_comb = [green_horn_comb,mr_market_classic]
        for comb in exist_comb:
            final_comb_exist = run_once_money_sim_market(comb,ans_new,'other',first_input_total,y,month,use=1,mode=4)
            compared_combs.append(final_comb_exist)

    elif market=='tw':
        labels = ['comb','market','black swan','grow-yuanta']
        exist_comb = [black_swan,grow_yuanta]
        for comb in exist_comb:
            final_comb_exist = run_once_money_sim_market(comb,ans_new,'other',first_input_total,y,month,use=1,mode=4)
            compared_combs.append(final_comb_exist)

    return final_input_money,final_sum_money,compared_combs,labels


# %%
# def plot_money_sim(filepath,y,month,market,first_input_total,ans,ans_new):
def plot_money_sim(filepath,final_input_money,final_sum_money,compared_combs,labels):
    
    # final_input_money,final_sum_money,compared_combs,labels = calculate_combs(y,month,market,first_input_total,ans,ans_new)

    fig_y_min = 60000
    fig_y_max = 600000

    reward = round((final_sum_money[-1]-final_input_money[-1])/final_input_money[-1],5)
    txt_comb = 'comb reward: '+str(reward)
    plt.plot(final_input_money, color='r',label='input')
    plt.plot(final_sum_money, color='b',label='comb')
    plt.text(0,fig_y_max-25000,txt_comb)

    colors = ['g','black','orange']
    for i in range(len(compared_combs)):
        comb = compared_combs[i]
        reward = comb[0].test_reward
        txt = labels[i+1] + ' reward: '+str(reward)
        plt.plot(comb[0].test_sum_money, color=colors[i],label=labels[i+1])
        plt.text(0,fig_y_max-25000*(i+2),txt)

    plt.ylim([fig_y_min,fig_y_max])
    # plt.legend(loc='lower right')
    plt.savefig(filepath+'compare2marketnew-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
    # plt.show()
    plt.clf()
    plt.close()

# %%
def cal_ann_reward_list(input_money,sum_money):

    reward_list = [0]
    input = input_money[0]
    count_month = 1
    for i in range(len(input_money)):
        if input_money[i]>input:
            r = (sum_money[i-1]-input)/input
            input = input_money[i]
            nnnn = count_month/12
            ann_r = math.pow( (1+r), 1/nnnn )-1
            reward_list.append(ann_r)
            count_month += 1
    
    r = (sum_money[i-1]-input)/input
    input = input_money[i]
    nnnn = count_month/12
    ann_r = math.pow( (1+r), 1/nnnn )-1
    reward_list.append(ann_r)

    return reward_list

# %%
def plot_ann_reward(filepath,final_input_money,final_sum_money,compared_combs,labels):

    fig_y_min = -1
    fig_y_max = 1.5

    reward_list_comb = cal_ann_reward_list(final_input_money,final_sum_money)

    # plt.plot(final_input_money, color='r',label='input')
    plt.plot(reward_list_comb, color='b',label='comb')
    for j in range(1,len(reward_list_comb)):
        txt = str(round(reward_list_comb[j],5))
        # plt.text(j-0.25,reward_list_comb[j]+0.005,txt)

    colors = ['g','black','orange']
    for i in range(len(compared_combs)):
        comb = compared_combs[i]
        reward_list = cal_ann_reward_list(final_input_money,comb[0].test_sum_money)
        plt.plot(reward_list, color=colors[i],label=labels[i+1])
        for j in range(1,len(reward_list)):
            txt = str(round(reward_list[j],5))
            # plt.text(j-0.25,reward_list[j]+0.005,txt)

    plt.ylim([fig_y_min,fig_y_max])
    # plt.legend(loc='lower left')
    plt.savefig(filepath+'ann_reward-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
    # plt.show()
    plt.clf()
    plt.close()

# %%
# def compare_plot_3D(filepath,market,y,month,ans_list,legends,comb_values,ans_new,first_input_total,mode=4,use=1):
def compare_plot_3D(filepath,comb_values,legends,compared_combs):
    # true_first_in = first_input_total
    rewards = [comb_values[0]]
    risks = [comb_values[2]]
    mdds = [comb_values[3]]
    for comb in compared_combs:
        rewards.append(comb[0].test_reward)
        risks.append(comb[0].test_risk)
        mdds.append(comb[0].test_mdd)

    # print(rewards,risks,mdds)
    color = ['b','g','black','orange','purple','r']
    if len(compared_combs)>0:
        p=[]
        ax = plt.subplot(projection='3d')
        for i in range(len(rewards)):
            ttt_fig = ax.scatter([risks[i]],[mdds[i]],[rewards[i]],  marker='o', s=40 ,c=color[i])
            p.append(ttt_fig)
        ax.set_zlabel('reward') 
        ax.set_ylabel('mdd')
        ax.set_xlabel('risk')
        ax.legend(p,legends)
        plt.savefig(filepath+'3Dfig-'+market+'-'+str(y)+'-'+str(month)+'.jpg')
        # plt.show()
        plt.clf()
    return rewards,risks,mdds

# %%
# def plot_3D(filepath,y,month,market,ans,ans_new,first_input_total,mode=4,use=1):
def plot_3D(filepath,final_input_money,final_sum_money,compared_combs,labels):
    comb_values = calculate_values(final_input_money,final_sum_money)
    legends = labels
    rewards,risks,mdds = compare_plot_3D(filepath,comb_values,legends,compared_combs)
    
    return legends,rewards,risks,mdds

# %%

def save_performance_metrixs(legends,rewards,risks,mdds):
    # print('performance metrixs:')
    # print('comb','reward','risk','mdd')
    df_list = []
    for i in range(len(legends)):
        tmp = [legends[i],rewards[i],risks[i],mdds[i]]
        df_list.append(tmp)
        # print(legends[i],rewards[i],risks[i],mdds[i])
        df = pd.DataFrame(df_list,columns=['comb','reward','risk','mdd'])
        df.to_csv(filepath+'performance-'+market+'-'+str(y)+'-'+str(month)+'.csv',index=False)



# %%

if __name__ == '__main__':
    
    # filepath = 'D:/Alia/Documents/asset allocation/plt/experiance/'
    (y,month,mode,use) = (2018,6,4,1)
    first_input_total = 150000
    market = 'us'
    # market = 'tw'
    
    # ans = [['IJJ SCC', '0.35051 0.64949']]
    # ans_new = [['IJJ SCC', '0.4 0.6'],['IJJ SCC', '0.5 0.5']]
    # # ans = [['LBJ TLH SOXS', '0.31229 0.57214 0.11557']]
    # # ans_new = [['LBJ TLH SOXS', '1.0 0.0 0.0'], ['LBJ TLH SOXS', '0.80148 0.09926 0.09926']]

    # final_input_money,final_sum_money,compared_combs,labels = calculate_combs(y,month,market,first_input_total,ans,ans_new)

    # plot_money_sim(filepath,final_input_money,final_sum_money,compared_combs,labels)
    # legends,rewards,risks,mdds = plot_3D(filepath,final_input_money,final_sum_money,compared_combs,labels)
    # plot_ann_reward(filepath,final_input_money,final_sum_money,compared_combs,labels)
    # save_performance_metrixs(legends,rewards,risks,mdds)
    # # print('performance metrixs:')
    # # print('comb','reward','risk','mdd')
    # # for i in range(len(legends)):
    # #     print(legends[i],rewards[i],risks[i],mdds[i])

##################### 批次處理 ######################
    paths = ['us-arima','us-ew','us-hw','us-lstm','us-mvp','us-mvtp']
    # paths = ['us-hw','us-mvp','us-mvtp']
    # paths = ['us-lstm']

    # filepath = 'D:/Alia/Documents/asset allocation/experience/fig/us-arima/'
    # ans_path = 'D:/Alia/Documents/asset allocation/experience/us-arima/'
    for p in paths:
        filepath = 'D:/Alia/Documents/asset allocation/output/performance/'+p+'/' # 輸出路徑
        ans_path = 'D:/Alia/Documents/asset allocation/output/answer/'+p+'/' # 預測好的答案放在哪裡
        allFileList = os.listdir(ans_path)
        # os.mkdir(filepath)
        for f in allFileList:
            print(f)
            ans_df = pd.read_csv(ans_path+f)
            y = ans_df['year'][0]
            month = ans_df['month'][0]
            ans = [[ans_df['names'][0],ans_df['weights'][0]]]
            ans_new = []
            for i in range(1,len(ans_df)):
                tmp = [ans_df['names'][i],ans_df['weights'][i]]
                ans_new.append(tmp)

            # print(ans)
            # print(ans_new)
            # 金流模擬計算
            final_input_money,final_sum_money,compared_combs,labels = calculate_combs(y,month,market,first_input_total,ans,ans_new)

            plot_money_sim(filepath,final_input_money,final_sum_money,compared_combs,labels) # 畫金流模擬圖
            legends,rewards,risks,mdds = plot_3D(filepath,final_input_money,final_sum_money,compared_combs,labels) # 畫3D圖
            plot_ann_reward(filepath,final_input_money,final_sum_money,compared_combs,labels) # 畫年化報酬率折線圖
            save_performance_metrixs(legends,rewards,risks,mdds) # 存performance metrix
            # break
        # break
 
# %%

# 改3績效陣列直接畫3D圖

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 2019
# rewards = [0.231335292,0.123116176,0.148591794,-0.633515762,-0.352292918,-0.608699261,-0.457315385,-0.252530441,0.834319362]
# risks = [0.1100082,0.099147772,0.101477716,0.432262565,0.408422573,0.508498651,0.490790835,0.272523568,1.107351434]
# mdds = [0.066183696,0.033392507,0.011268708,0.539668852,0.234817269,0.572911984,0.440476231,0.121090092,0.25582934]

# 2018
# rewards = [-0.064528885,-0.073095678,-0.048498329,-0.14114819,-0.143104262,-0.429474686,-0.338055962,-0.052034576,1.542250588]
# risks = [0.216523842,0.163698105,0.180279605,0.574320523,0.228191091,0.683037616,0.919984106,0.130735373,1.921460885]
# mdds = [0.193489279,0.108768209,0.050098198,0.402846927,0.198522152,0.525458204,0.65073466,0.079838679,0.149457714]

# 2017 
# rewards = [0.163004115,0.135187909,0.105176024,-0.252891038,-0.185256902,-0.258836396,0.093137374,-0.128705062,0.655650251]
# risks = [0.130836292,0.123528961,0.128597201,0.48071039,0.427365528,0.501080707,0.742602296,0.366616951,0.995989103]
# mdds = [0.026101078,0.010925195,0.014687046,0.53952504,0.364468394,0.417539636,0.620339007,0.271787552,0.337393586]

# 2016
rewards = [0.113046796,0.039380095,0.059770655,-0.474030078,-0.546421224,-0.42258001,-0.748431954,-0.46193478,0.286031473]
risks = [0.141495174,0.123107411,0.114080307,0.818061878,0.655343493,0.65940516,0.80436484,0.430423855,0.614665822]
mdds = [0.091875236,0.030559173,0.037839126,0.71283552,0.675396285,0.62630015,0.895372236,0.395598903,0.444561753]

legends = ['market','green horn','mr.market','arima','equal weight','holt winter','lstm','mvp','mvtp']
color = ['green','black','orange','purple','red','yellow','blue','gray','yellowgreen']
p=[]
ax = plt.subplot(projection='3d')
for i in range(len(rewards)):
    ttt_fig = ax.scatter([risks[i]],[mdds[i]],[rewards[i]],  marker='o', s=40 ,c=color[i])
    p.append(ttt_fig)
ax.set_zlabel('reward') 
ax.set_ylabel('mdd')
ax.set_xlabel('risk')
# ax.legend(p,legends,loc='lower right')
plt.savefig('D:/Alia/Documents/asset allocation/output/performance/all/3Dfig-us-2016-1.jpg')
plt.show()
plt.clf()

# %%

# ans = [['IJJ SCC', '0.35051 0.64949'],
# ['IJJ UGA SCC SZK', '0.17525 0.17525 0.32475 0.32475'],
# ['IJJ UGA ESGU SCC SZK DRV', '0.11684 0.11684 0.11684 0.2165 0.2165 0.2165'],
# ['IJJ UGA ESGU RDIV SCC SZK DRV SDP', '0.08763 0.08763 0.08763 0.08763 0.16237 0.16237 0.16237 0.16237'],
# ['IJJ UGA ESGU RDIV SHE SCC SZK DRV SDP XES', '0.0701 0.0701 0.0701 0.0701 0.0701 0.1299 0.1299 0.1299 0.1299 0.1299'],
# ['IJJ UGA ESGU RDIV SHE IAT SCC SZK DRV SDP XES ERX', '0.05842 0.05842 0.05842 0.05842 0.05842 0.05842 0.10825 0.10825 0.10825 0.10825 0.10825 0.10825'],
# ['IJJ UGA ESGU RDIV SHE IAT XSD SCC SZK DRV SDP XES ERX GDXJ', '0.05007 0.05007 0.05007 0.05007 0.05007 0.05007 0.05007 0.09278 0.09278 0.09278 0.09278 0.09278 0.09278 0.09278'],
# ]

# ans = [['006208.TW 00674R.TW 00668K.TW', '0.06267 0.47256 0.46477']]
# ans = [ ['0050.TW 00642U.TW 00661.TW', '0.299 0.45907 0.24193'],
# ['0050.TW 006204.TW 00642U.TW 00674R.TW 00661.TW 00641R.TW', '0.1495 0.1495 0.22954 0.22954 0.12096 0.12096'],
# ['0050.TW 006204.TW 0052.TW 00642U.TW 00674R.TW 00638R.TW 00661.TW 00641R.TW 00669R.TW', '0.09967 0.09967 0.09967 0.15302 0.15302 0.15302 0.08064 0.08064 0.08064'],
# ['0050.TW 006204.TW 0052.TW 0057.TW 00642U.TW 00674R.TW 00638R.TW 00675L.TW 00661.TW 00641R.TW 00669R.TW 00657.TW', '0.07475 0.07475 0.07475 0.07475 0.11477 0.11477 0.11477 0.11477 0.06048 0.06048 0.06048 0.06048'] ]
# ['0050.TW 006204.TW 0052.TW 0057.TW 006208.TW 00642U.TW 00674R.TW 00638R.TW 00675L.TW 00654R.TW 00661.TW 00641R.TW 00669R.TW 00657.TW 00657K.TW', '0.0598 0.0598 0.0598 0.0598 0.0598 0.09181 0.09181 0.09181 0.09181 0.09181 0.04839 0.04839 0.04839 0.04839 0.04839']]


# def run_once(y,month,mode,use,ans,nochange):

#     # ans = choose.choose_target(y)
#     # ans = [['006208.TW 00674R.TW 00668K.TW', '0.06267 0.47256 0.46477']]
#     # ans = [['SPY', '1.0']]
#     print(ans)
#     if len(ans)==0:
#         return [],[],[]
#     comb_list,gotest_candidate = train_evaluation(y,month,ans,mode)
#     gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total,nochange)
#     # final_comb[0].print_test_evalu()
#     # print(final_comb[0].test_sum_money)
#     # plt.plot(final_comb[0].test_sum_money, color='b')
#     # plt.plot(final_comb[0].input_money, color='r')
#     # plt.show()
#     # print(final_comb[0].test_sum_money[20],final_comb[0].input_money[20])
#     return gotest_candidate,final_ans_candidate,final_comb

# def compare2market(filepath,y,month,mode,use,market,filename,ans,first_input_total):
#     if market == 'tw':
#         ans_market = [twii] # 大盤
#     elif market == 'us':
#         ans_market = [spy] # 大盤
#     comb_list,gotest_candidate = train_evaluation(y,month,ans_market,mode)
#     gotest_candidate_market,final_ans_candidate_market,final_comb_market = test_evaluation(y,month,gotest_candidate,use,mode,first_input_total)
        
#     gotest_candidate,final_ans_candidate,final_comb = run_once(y,month,mode,use,ans)
    
#     index = 0
#     first_input = final_comb[0].input_money[0]
#     for i in range(len(final_comb[0].test_sum_money)):
#         if final_comb[0].input_money[i]==first_input:
#             index = i
#         else:
#             break
#     month_r_comb = round((final_comb[0].test_sum_money[index]-first_input)/first_input,5)
#     month_r_market = round((final_comb_market[0].test_sum_money[index]-first_input)/first_input,5)
#     txt_comb = 'comb 1 month reward: '+str(month_r_comb)
#     txt_market = 'market 1 month reward: '+str(month_r_market)
#     # txt_comb = 'comb reward: '+str(round(final_comb[0].test_ann_reward,5))+', 1month reward: '+str(month_r_comb)
#     # txt_market = 'market reward: '+str(round(final_comb_market[0].test_ann_reward,5))+', 1month reward: '+str(month_r_market)

#     plt.plot(final_comb[0].test_sum_money, color='b',label='comb')
#     plt.plot(final_comb_market[0].test_sum_money, color='g',label='market')
#     plt.plot(final_comb[0].input_money, color='r',label='input')
#     plt.ylim([120000,180000])
#     plt.legend(loc='lower right')
#     plt.text(0,175000,txt_comb)
#     plt.text(0,170000,txt_market)
#     plt.savefig(filepath+filename)
#     # plt.show()
#     plt.clf()