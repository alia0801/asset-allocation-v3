# %%
# import choose
import evaluation_rebalance
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
# import os

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
def train_evaluation(y,month,ans,mode):
    
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
    
        evaluation_value = evaluation_rebalance.train_choose(y_train,month,choose_name,weight,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
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
def test_evaluation(y,month,gotest_candidate,use,mode):
    nnnn = 1
    y_test = y
    for a in range(len(gotest_candidate)):

        tmp = gotest_candidate[a]
        choose_name = tmp.the_names
        weight = tmp.the_weight

        evaluation_value = evaluation_rebalance.test_choose(y_test,month,choose_name,weight,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        
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
def run_once(y,month,mode,use,ans):

    # ans = choose.choose_target(y)
    # ans = [['006208.TW 00674R.TW 00668K.TW', '0.06267 0.47256 0.46477']]
    # ans = [['SPY', '1.0']]
    print(ans)
    if len(ans)==0:
        return [],[],[]
    comb_list,gotest_candidate = train_evaluation(y,month,ans,mode)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode)
    # final_comb[0].print_test_evalu()
    # print(final_comb[0].test_sum_money)
    # plt.plot(final_comb[0].test_sum_money, color='b')
    # plt.plot(final_comb[0].input_money, color='r')
    # plt.show()
    # print(final_comb[0].test_sum_money[20],final_comb[0].input_money[20])
    return gotest_candidate,final_ans_candidate,final_comb

# %%
def compare2market(filepath,y,month,mode,use,market,filename,ans):
    if market == 'tw':
        ans_market = [twii] # 大盤
    elif market == 'us':
        ans_market = [spy] # 大盤
    comb_list,gotest_candidate = train_evaluation(y,month,ans_market,mode)
    gotest_candidate_market,final_ans_candidate_market,final_comb_market = test_evaluation(y,month,gotest_candidate,use,mode)
        
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,month,mode,use,ans)
    
    index = 0
    first_input = final_comb[0].input_money[0]
    for i in range(len(final_comb[0].test_sum_money)):
        if final_comb[0].input_money[i]==first_input:
            index = i
        else:
            break
    month_r_comb = round((final_comb[0].test_sum_money[index]-first_input)/first_input,5)
    month_r_market = round((final_comb_market[0].test_sum_money[index]-first_input)/first_input,5)
    txt_comb = 'comb 1 month reward: '+str(month_r_comb)
    txt_market = 'market 1 month reward: '+str(month_r_market)
    # txt_comb = 'comb reward: '+str(round(final_comb[0].test_ann_reward,5))+', 1month reward: '+str(month_r_comb)
    # txt_market = 'market reward: '+str(round(final_comb_market[0].test_ann_reward,5))+', 1month reward: '+str(month_r_market)

    plt.plot(final_comb[0].test_sum_money, color='b',label='comb')
    plt.plot(final_comb_market[0].test_sum_money, color='g',label='market')
    plt.plot(final_comb[0].input_money, color='r',label='input')
    plt.ylim([120000,180000])
    plt.legend(loc='lower right')
    plt.text(0,175000,txt_comb)
    plt.text(0,170000,txt_market)
    plt.savefig(filepath+filename)
    # plt.show()
    plt.clf()

# def run_bigtable(y,month,mode,use):
#     # ans = [['006208.TW 00674R.TW 00668K.TW', '0.06267 0.47256 0.46477']]
#     # ans = choose.choose_target(y)
#     for i in range(len(ans)):
#         if len(ans)==0:
#             continue
#         comb_list,gotest_candidate = train_evaluation(y,month,ans,mode)
#         gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,month,gotest_candidate,use,mode)
#         df_list = []
#         for b in range(len(gotest_candidate)):
#             # gotest_candidate[b].print_test_evalu()
#             tmp = gotest_candidate[b]

#             w = []
#             for a in range(len(tmp.the_weight)):
#                 w.append(str(tmp.the_weight[a]))

#             tmp_list = [y,month,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd]
#             df_list.append(tmp_list)
#             # big_list.append(tmp_list)
#         df = pd.DataFrame(df_list,columns=['year','month','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd'])
#         # df.to_csv(filepath+'big_table/'+ str(y)+'_' + str(expect_reward) +'.csv',index=False)
#         # break
# %%
def compare_plot_3D(filepath,y,month,ans_list,legends,mode=4,use=1):
    rewards = []
    risks = []
    mdds = []
    for ans in ans_list:
        choose = ans[0].split(' ')
        w = ans[1].split(' ')
        weight = []
        for i in range(len(w)):
            weight.append(float(w[i]))
        values = evaluation_rebalance.test_choose(y,month,choose,weight,mode)
        rewards.append(values[0])
        risks.append(values[2])
        mdds.append(values[3])
    # print(rewards,risks,mdds)
    color = ['g','b','black','r','orange','purple']
    if len(ans_list)>0:
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
# %%
def plot_3D(filepath,y,month,market,ans,mode=4,use=1):
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,month,mode,use,ans)

    tmp_w = []
    for i in range(len(final_comb[0].the_weight)):
        tmp_w.append(str(final_comb[0].the_weight[i]))
    tmp_ans = [' '.join(final_comb[0].the_names),' '.join(tmp_w)]
    if market=='us':
        ans_list = [spy,tmp_ans,green_horn_comb,mr_market_classic]
        legends = ['market','comb','green_horn','mr.market']
    else:
        ans_list = [twii,tmp_ans,black_swan,grow_yuanta]
        legends = ['market','comb','black_swan','grow_yuanta']
    # ans_list = [spy]
    # ans_list += ans
    compare_plot_3D(filepath,y,month,ans_list,legends,mode=4,use=1)


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

if __name__ == '__main__':
    
    filepath = 'D:/Alia/Documents/asset allocation/plt/experiance/'
    y = 2018
    month = 6
    mode = 4
    use = 1
    # market = 'us'
    market = 'tw'
    # filename = 'compare2market-'+market+'-'+str(y)+'-'+str(month)+'.jpg'
    
    # ans_path = 'D:/Alia/Documents/asset allocation/experiance/'
    # allFileList = os.listdir(ans_path)
    # for f in allFileList:
    #     print(f)
    #     ans_df = pd.read_csv(ans_path+f)
    #     y = ans_df['year'][0]
    #     month = ans_df['month'][0]
    #     ans = []
    #     for i in range(len(ans_df)):
    #         name = ans_df['names'][i]
    #         w = ans_df['weights'][i]
    #         ans.append([name,w])
    #     # print(ans)
    #     # break

    #     filename = 'compare2market-'+market+'-'+str(y)+'-'+str(month)+'.jpg'

    #     compare2market(filepath,y,month,mode,use,market,filename,ans)

    #     plot_3D(filepath,y,month,market,ans,mode,use)

        # gotest_candidate,final_ans_candidate,final_comb = run_once(2018,6,4,1)

        # tmp_w = []
        # for i in range(len(final_comb[0].the_weight)):
        #     tmp_w.append(str(final_comb[0].the_weight[i]))
        # tmp_ans = [' '.join(final_comb[0].the_names),' '.join(tmp_w)]
        # ans_list = [spy,tmp_ans]
        # # ans_list = [spy]
        # # ans_list += ans
        # compare_plot_3D(filepath,ans_list,mode=4,use=1)

# %%
