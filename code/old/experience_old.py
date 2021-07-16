# %%
import choose
# import choose_olddb
# import skyline_choose
# import raw_choose
# import slope_choose
import evaluation_rebalance
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

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
        print('test_risk',self.test_risk)
        print('test_mdd',self.test_mdd)
        print('success_ratio',self.success_ratio)   
        print('stat_list',self.stat_list)
        print('success_stat',self.success_stat)      

# %%

tw50 = [0,0,0,'0050.TW','1.0']
twii = [0,0,0,'006204.TW','1.0']
spy = [0,0,0,'SPY','1.0']
black_swan = [0, 0, 0, '00679B.TW 00635U.TW 00682U.TW 00706L.TW','0.2491 0.3086 0.217 0.2253' ] # black swan
grow_yuanta = [0, 0, 0, '00646.TW 0050.TW 006206.TW 00661.TW 00660.TW 00697B.TW 00720B.TW 00635U.TW 00682U.TW', '0.1791 0.2301 0.1082 0.0747 0.0458 0.1150 0.0271 0.0782 0.1418'] # 成長
green_horn_comb = [0,0,0,'VTI VGK VPL VWO BWX IEI','0.15 0.15 0.15 0.15 0.2 0.2']
mr_market_classic = [0,0,0,'ITOT VEU VNQ AGG','0.36 0.18 0.06 0.4']

evaluation_value_list = ['reward','annual_reward','v1_std','mdd','success_ratio','success']

# %%
# train evaluation
# train y-n/1/1 ~ y-1/12/31
def train_evaluation(y,nnnn,ans,expect_reward,mode):
    
    y_train = y-nnnn
    comb_list = []
    for a in range(len(ans)):
        choose1 = ans[a][3]
        weight1 = ans[a][4]
    
        choose_name = choose1.split(' ')
        weight = weight1.split(' ')
        for i in range(len(weight)):
            weight[i] = float(weight[i])
    
        evaluation_value = evaluation_rebalance.test_choose(y_train,nnnn,choose_name,weight,expect_reward,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        if evaluation_value is None:
            continue
        # success_ratio,success,stat_list,stat = evaluation_rebalance.slidwindow(y_train,nnnn,choose_name,weight,expect_reward,mode)
        # evaluation_value.append(success_ratio)
        # evaluation_value.append(success)
        # evaluation_value.append(stat_list)
        # evaluation_value.append(stat)
        
        tmp = Ans_comb()
        tmp.the_names = choose_name
        tmp.the_weight = weight
        tmp.train_reward = evaluation_value[0]
        tmp.train_ann_reward = evaluation_value[1]
        tmp.train_risk = evaluation_value[2]
        tmp.train_mdd = evaluation_value[3]
        # tmp.success_ratio = evaluation_value[6]
        # tmp.success = evaluation_value[7]
        # tmp.success_stat = evaluation_value[8]
        # tmp.stat_list = evaluation_value[9]
    
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
# test y/1/1 ~ y+nnnn-1/12/31
# test evaluation

def test_evaluation(y,nnnn,gotest_candidate,use,expect_reward,mode):
    y_test = y
    for a in range(len(gotest_candidate)):

        tmp = gotest_candidate[a]
        choose_name = tmp.the_names
        weight = tmp.the_weight

        evaluation_value = evaluation_rebalance.test_choose(y_test,nnnn,choose_name,weight,expect_reward,mode) # reward,annual_reward,v1_std,mdd,success_ratio,success
        
        success_ratio,success,stat_list,stat = evaluation_rebalance.slidwindow(y_test,nnnn,choose_name,weight,expect_reward,mode)
        evaluation_value.append(success_ratio)
        evaluation_value.append(success)
        evaluation_value.append(stat_list)
        evaluation_value.append(stat)
        
        # if evaluation_value is None:
            # evaluation_value = [-100,-100,100,100,[]]
        tmp.test_reward = evaluation_value[0]
        tmp.test_ann_reward = evaluation_value[1]
        tmp.test_risk = evaluation_value[2]
        tmp.test_mdd = evaluation_value[3]
        tmp.test_sum_money = evaluation_value[4]
        tmp.input_money = evaluation_value[5]
        tmp.success_ratio = evaluation_value[6]
        tmp.success = evaluation_value[7]
        tmp.success_stat = evaluation_value[8]
        tmp.stat_list = evaluation_value[9]


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
            if( (tmp_1.test_ann_reward >= tmp_2.test_ann_reward) 
                and (tmp_1.test_risk <= tmp_2.test_risk) 
                and (tmp_1.test_mdd <= tmp_2.test_mdd) 
                and (tmp_1.success_ratio >= tmp_2.success_ratio) ):
                # final_ans_candidate[b] = tmp_1
                # go = False
                # break
                go = True
                bad.append(b)
            elif( (tmp_1.test_ann_reward <= tmp_2.test_ann_reward) 
                and (tmp_1.test_risk >= tmp_2.test_risk) 
                and (tmp_1.test_mdd >= tmp_2.test_mdd) 
                and (tmp_1.success_ratio <= tmp_2.success_ratio) ):
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
                if tmp.test_ann_reward == final_ann_reward:
                    # print(tmp.test_ann_reward,final_ann_reward)
                    final_comb.append(tmp)
                elif tmp.test_ann_reward > final_ann_reward: 
                    final_comb = [tmp]
                    final_ann_reward = tmp.test_ann_reward
            # print(final_ann_reward)
        elif use == 1:
            # use test_risk
            final_risk = 1000
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                if tmp.test_risk == final_risk and tmp.test_ann_reward >= expect_reward:
                    final_comb.append(tmp)
                elif tmp.test_risk < final_risk and tmp.test_ann_reward >= expect_reward:
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
        elif use == 3:
            # use success_ratio
            final_success_ratio = 0
            final_comb=[]
            for a in range(len(final_ans_candidate)):
                tmp = final_ans_candidate[a]
                if tmp.success_ratio == final_success_ratio:
                    final_comb.append(tmp)
                elif tmp.success_ratio > final_success_ratio:    
                    final_comb = [tmp]  
                    final_success_ratio = tmp.success_ratio
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
# 跑一次選股+train評估篩選+test評估篩選，最後產出一個組合
def run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward):

    ans = raw_choose.choose_target(y,expect_reward,train_nnnn)
    print(ans)
    if len(ans)==0:
        return [],[],[]
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    # final_comb[0].print_test_evalu()
    # print(final_comb[0].test_sum_money)
    # plt.plot(final_comb[0].test_sum_money, color='b')
    # plt.plot(final_comb[0].input_money, color='r')
    # plt.show()
    return gotest_candidate,final_ans_candidate,final_comb


# %%
# train y-n/1/1 ~ y-1/12/31
# test y/1/1 ~ y+nnnn-1/12/31

# y = 2016 #train 2016~2017 
# 跑大表，畫skyline用
def run_big_table(filepath,train_nnnn=3,test_nnnn=3,mode=3,use=1,yyyys=[2012,2014,2016]):
    for y in yyyys:
        # 依據expect_reward選出所有可能組合、對所有組合進行train與test的2次評估並輸出成csv檔
        big_list = []
        for expect_reward in range(5,11):
            expect_reward /=100
            print(expect_reward)

            ans = raw_choose.choose_target(y,expect_reward,train_nnnn)

            print(ans)

            if len(ans)==0:
                continue
            comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans,expect_reward,mode)

            gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,test_nnnn,comb_list,use,expect_reward,mode)

            df_list = []
            for b in range(len(gotest_candidate)):
                # gotest_candidate[b].print_test_evalu()
                tmp = gotest_candidate[b]
                w = []

                for a in range(len(tmp.the_weight)):
                    w.append(str(tmp.the_weight[a]))
                st =[]
                for a in range(len(tmp.stat_list)):    
                    st.append(str(round(tmp.stat_list[a],5)))

                tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
                df_list.append(tmp_list)
                big_list.append(tmp_list)

            df = pd.DataFrame(df_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list'])
            df.to_csv(filepath+'big_table/'+ str(y)+'_' + str(expect_reward) +'.csv',index=False)
            # break
        df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list'])
        df_big.to_csv(filepath+'big_table/'+ str(y)+'_all.csv',index=False)
        # break
  
# %%
# 跑已知組合的大表
def run_SPY(filepath,filename,etf_name,ans_weight,train_nnnn=3,test_nnnn=5,mode=3,use=1,y1=2012,y2=2019):
    # nnnn = 3
    # mode = 3 #rebalance mode
    # use = 1
    big_list = []
    for expect_reward in range(3,15):
        expect_reward /= 100
        SPY_list=[]
        for y in range(y1,y2):
            # y = 2018

            # ans = [[0, 0, 0, 'SPY', '1.0']]  # 美股大盤
            # ans = [[0, 0, 0, '0050.TW', '1.0']]  # 台股大盤
            # ans = [[0, 0, 0, '006204.TW', '1.0']]  # 台股大盤
            ans = [[0, 0, 0, etf_name, ans_weight]]
            comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans,expect_reward,mode)
            gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)

            tmp = final_comb[0]
            w = []
            for a in range(len(tmp.the_weight)):
                w.append(str(tmp.the_weight[a]))
            st =[]
            for a in range(len(tmp.stat_list)):    
                st.append(str(round(tmp.stat_list[a],5)))
            tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
            SPY_list.append(tmp_list)
            big_list.append(tmp_list)
            # print(tmp_list)

        df = pd.DataFrame(SPY_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list'])
        df.to_csv(filepath+filename+'_'+str(expect_reward)+'.csv',index=False)

    df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list'])
    df_big.to_csv(filepath+filename+'_all.csv',index=False)


# %%

#############################跑一次要拿來比較的組合在train&test期間的值##############################################

def run_SPY_once(y,train_nnnn,test_nnnn,mode,use,expect_reward):
    ans = [[0, 0, 0, 'SPY', '1.0']]  # 美股大盤
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    return final_comb

def run_UScompare_once(y,train_nnnn,test_nnnn,mode,use,expect_reward):
    ans_spy = [spy]  # 美股大盤
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_spy,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_spy = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    ans_green = [green_horn_comb] # 綠角
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_green,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_green = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    ans_classic = [mr_market_classic]# 市場先生-經典-穩定
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_classic,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_classic = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    return final_comb_spy,final_comb_green,final_comb_classic    



def run_TW_once(y,train_nnnn,test_nnnn,mode,use,expect_reward):
    ans_twii = [twii] # 積極
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_twii,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    return final_comb
    
def run_TWcompare_once(y,train_nnnn,test_nnnn,mode,use,expect_reward):
    ans_black = [[0, 0, 0, '00679B.TW 00635U.TW 00682U.TW 00706L.TW','0.2491 0.3086 0.217 0.2253' ]] # black swan
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_black,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_black = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    ans_twii = [[0, 0, 0, '006204.TW', '1.0']] # 大盤
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_twii,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_twii = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    ans_stable = [[0, 0, 0, '00646.TW 0050.TW 006206.TW 00661.TW 00660.TW 00697B.TW 00720B.TW 00635U.TW 00682U.TW', '0.1791 0.2301 0.1082 0.0747 0.0458 0.1150 0.0271 0.0782 0.1418']]# 成長
    comb_list,gotest_candidate = train_evaluation(y,train_nnnn,ans_stable,expect_reward,mode)
    gotest_candidate,final_ans_candidate,final_comb_stable = test_evaluation(y,test_nnnn,gotest_candidate,use,expect_reward,mode)
    
    return final_comb_black,final_comb_twii,final_comb_stable

################################################## 畫圖 ###########################################

### 年化報酬率折線
def compare_reward_plot(filepath,ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode=4,use=1):
    rewards = []
    for ans in ans_list:
        choose = ans[3].split(' ')
        w = ans[4].split(' ')
        weight = []
        for i in range(len(w)):
            weight.append(float(w[i]))
        ann_r = evaluation_rebalance.get_rewards(y=y,nnnn=test_nnnn,choose=choose,weight=weight,mode=4)
        rewards.append(ann_r)
    color = ['g','b','orange','black','r','purple']
    xlabel = [y]
    for i in range(1,test_nnnn):
        xlabel.append(y+i)
    if len(ans_list)>0:
        p=[]
        for i in range(len(rewards)):
            ann_r = rewards[i]
            while len(ann_r)!=len(xlabel):
                ann_r.append(0)
            temp, = plt.plot(xlabel,ann_r,color=color[i],marker='o')
            p.append(temp)
            for j in range(len(ann_r)):
                plt.text(xlabel[j],ann_r[j]+0.001,'%.4f'%ann_r[j])
        plt.legend(p,legends)
        plt.ylim([-0.15,0.3])
        plt.savefig(filepath+'compare_r_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        # plt.show()
        plt.clf()    

### 畫3D圖

def compare_plot_3D(filepath,ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode=4,use=1):
    rewards = []
    risks = []
    mdds = []
    for ans in ans_list:
        choose = ans[3].split(' ')
        w = ans[4].split(' ')
        weight = []
        for i in range(len(w)):
            weight.append(float(w[i]))
        # values = evaluation_rebalance.get_rewards(y=y,nnnn=test_nnnn,choose=choose,weight=weight,mode=4)
        values = evaluation_rebalance.get_all_value(y=y,nnnn=test_nnnn,choose=choose,weight=weight,mode=4)
        rewards.append(values[1])
        risks.append(values[2])
        mdds.append(values[3])
    color = ['g','b','orange','black','r','purple']
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
        plt.savefig(filepath+'3d_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        # plt.show()
        plt.clf()  


### 金流模擬圖(+大盤)
def US_plot_sim(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward):
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    SPY_comb = run_SPY_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    if len(final_comb)>0:
        temp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        temp2, = plt.plot(SPY_comb[0].test_sum_money, color='g')
        temp3, = plt.plot(final_comb[0].input_money, color='r')
        p = [temp1,temp2,temp3]
        legends = ['our_comb','SPY','input_money']
        plt.legend(p,legends)
        plt.ylim([120000,1500000])
        # plt.show()
        plt.savefig(filepath+'US_sim_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        plt.clf()
    return final_comb

### 金流模擬圖(+大盤+比較)
def US_plot_sim_compare(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward):
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    comb_spy,comb_green,comb_classic = run_UScompare_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    # SPY_comb = run_SPY_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    if len(final_comb)>0:
        temp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        temp2, = plt.plot(comb_classic[0].test_sum_money, color='black')
        temp3, = plt.plot(comb_spy[0].test_sum_money, color='g')
        temp4, = plt.plot(comb_green[0].test_sum_money, color='orange')
        temp5, = plt.plot(final_comb[0].input_money, color='r')
        p = [temp1,temp2,temp3,temp4,temp5]
        legends = ['our_comb','Mr.market_classic','SPY','green_horn','input_money']
        plt.legend(p,legends)
        plt.ylim([120000,1500000])
        # plt.show()
        plt.savefig(filepath+'fig_US_sim_comp/'+'US_sim_comp_'+str(y)+'_'+str(expect_reward)+'.jpg')
        plt.clf()
        
        temp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        temp2, = plt.plot(comb_spy[0].test_sum_money, color='g')
        temp3, = plt.plot(final_comb[0].input_money, color='r')
        p = [temp1,temp2,temp3]
        legends = ['our_comb','SPY','input_money']
        plt.legend(p,legends)
        plt.ylim([120000,1500000])
        # plt.show()
        plt.savefig(filepath+'fig_US_sim/'+'US_sim_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        plt.clf()
    return final_comb
   
### 金流模擬圖(+大盤)
def TW_plot_sim(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward):
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    final_comb_twii = run_TW_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    if len(final_comb)>0:
        tmp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        tmp2, = plt.plot(final_comb_twii[0].test_sum_money, color='g')
        tmp3, = plt.plot(final_comb[0].input_money, color='r')
        p = [tmp1,tmp2,tmp3]
        legends = ['our_comb','twii','input']
        plt.legend(p,legends)
        plt.ylim([100000,850000])
        plt.savefig(filepath+'fig_TW_sim/'+'TW_sim_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        # plt.show()
        plt.clf()
    return final_comb

### 金流模擬圖(+大盤+比較)
def TW_plot_sim_compare(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward):
    gotest_candidate,final_ans_candidate,final_comb = run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    final_comb_black,final_comb_twii,final_comb_stable = run_TWcompare_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
    if len(final_comb)>0:
        tmp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        tmp2, = plt.plot(final_comb_black[0].test_sum_money, color='black')
        tmp3, = plt.plot(final_comb_twii[0].test_sum_money, color='g')
        tmp4, = plt.plot(final_comb_stable[0].test_sum_money, color='orange')
        tmp5, = plt.plot(final_comb[0].input_money, color='r')
        p = [tmp1,tmp2,tmp3,tmp4,tmp5]
        legends = ['our_comb','black swan','twii','grow','input']
        plt.legend(p,legends)
        plt.ylim([100000,850000])
        plt.savefig(filepath+'fig_TW_sim_comp/'+'TW_sim_comp_'+str(y)+'_'+str(expect_reward)+'.jpg')
        # plt.show()
        plt.clf()

        tmp1, = plt.plot(final_comb[0].test_sum_money, color='b')
        tmp2, = plt.plot(final_comb_twii[0].test_sum_money, color='g')
        tmp3, = plt.plot(final_comb[0].input_money, color='r')
        p = [tmp1,tmp2,tmp3]
        legends = ['our_comb','twii','input']
        plt.legend(p,legends)
        plt.ylim([100000,850000])
        plt.savefig(filepath+'fig_TW_sim/'+'TW_sim_fig_'+str(y)+'_'+str(expect_reward)+'.jpg')
        # plt.show()
        plt.clf()
    return final_comb


######################## 畫圖-與大盤 & 輸出選出之csv #########################
def run_plot_US(filepath,train_nnnn=3,test_nnnn=3,mode=4,use=1,ab = [5,6,9,10,12,13],y1=2012,y2=2017):
    big_list = []
    for expect_reward in ab:
        expect_reward /= 100
        SPY_compare = []
        for y in range(y1,y2):
            final_comb = US_plot_sim(filepath+'fig_US_sim/',y,train_nnnn,test_nnnn,mode,use,expect_reward)
            
            if len(final_comb)>0:
                tmp = final_comb[0]
                w = []
                for a in range(len(tmp.the_weight)):
                    w.append(str(tmp.the_weight[a]))
                st =[]
                for a in range(len(tmp.stat_list)):    
                    st.append(str(round(tmp.stat_list[a],5)))
                tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
                
                ans_list = [spy]
                ans_list.append([0,0,0,' '.join(tmp.the_names), ' '.join(w)])
                legends = ['SPY','our_comb']
                compare_reward_plot(filepath+'fig_US_r/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)

                SPY_compare.append(tmp_list)
                big_list.append(tmp_list)

            else:
                SPY_compare.append([])
                big_list.append([])
            # break
        if len(SPY_compare[0])>0:
            df = pd.DataFrame(SPY_compare,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
            df.to_csv(filepath+'US_compare_'+str(expect_reward)+'.csv',index=False)
        # break

    df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
    df_big.to_csv(filepath+'US_compare_all.csv',index=False)

def run_plot_TW(filepath,train_nnnn=3,test_nnnn=3,mode=4,use=1,ab = [5,6,9,10,12,13],y1=2012,y2=2017):
    big_list = []
    for expect_reward in ab:
        expect_reward /= 100
        SPY_compare = []
        for y in range(y1,y2):
            final_comb = TW_plot_sim(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward)
            
            if len(final_comb)>0:
                tmp = final_comb[0]
                w = []
                for a in range(len(tmp.the_weight)):
                    w.append(str(tmp.the_weight[a]))
                st =[]
                for a in range(len(tmp.stat_list)):    
                    st.append(str(round(tmp.stat_list[a],5)))
                tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
                ans_list = [twii]
                
                ans_list.append([0,0,0,' '.join(tmp.the_names), ' '.join(w)])
                legends = ['006204.TW','our_comb']
                compare_reward_plot(filepath+'fig_TW_r/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)

                SPY_compare.append(tmp_list)
                big_list.append(tmp_list)

            else:
                SPY_compare.append([])
                big_list.append([])
            # break
        if len(SPY_compare[0])>0:
            df = pd.DataFrame(SPY_compare,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
            df.to_csv(filepath+'TW_compare_'+str(expect_reward)+'.csv',index=False)
        # break

    df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
    df_big.to_csv(filepath+'TW_compare_all.csv',index=False)


#################### 畫圖-大盤&現有組合 & 輸出選出之csv ####################
def run_plot_TW_compare(filepath,train_nnnn=3,test_nnnn=3,mode=4,use=1,ab = [5,6,9,10,12,13],y1=2012,y2=2017):
    big_list = []
    for expect_reward in ab:
        expect_reward /= 100
        SPY_compare = []
        for y in range(y1,y2):
            # y = 2018
            final_comb = TW_plot_sim_compare(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward)
            
            if len(final_comb)>0:
                tmp = final_comb[0]
                w = []
                for a in range(len(tmp.the_weight)):
                    w.append(str(tmp.the_weight[a]))
                st =[]
                for a in range(len(tmp.stat_list)):    
                    st.append(str(round(tmp.stat_list[a],5)))
                tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
                ans_list = [twii]
                
                ans_list.append([0,0,0,' '.join(tmp.the_names), ' '.join(w)])
                legends = ['006204.TW','our_comb']
                compare_reward_plot(filepath+'fig_TW_r/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)
                
                
                ans_list.append(grow_yuanta)
                ans_list.append(black_swan)
                legends = ['006204.TW','our_comb','grow','black_swan']
                compare_reward_plot(filepath+'fig_TW_r_comp/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)
                compare_plot_3D(filepath+'fig_TW_3D_comp/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)

                SPY_compare.append(tmp_list)
                big_list.append(tmp_list)

            else:
                SPY_compare.append([])
                big_list.append([])
            # break
        if len(SPY_compare[0])>0:
            df = pd.DataFrame(SPY_compare,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
            df.to_csv(filepath+'TW_compare_'+str(expect_reward)+'.csv',index=False)
            # break
    df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
    df_big.to_csv(filepath+'TW_compare_all.csv',index=False)

def run_plot_US_compare(filepath,train_nnnn=3,test_nnnn=5,mode=4,use=1,ab = [5,6,9,10,12,13],y1=2012,y2=2017):
    big_list = []
    for expect_reward in ab:
        expect_reward /= 100
        SPY_compare = []
        for y in range(y1,y2):
            # y = 2018
            final_comb = US_plot_sim_compare(filepath,y,train_nnnn,test_nnnn,mode,use,expect_reward)
            # gotest_candidate,final_ans_candidate,final_comb = run_once(y,train_nnnn,test_nnnn,mode,use,expect_reward)
            
            if len(final_comb)>0:
                tmp = final_comb[0]
                w = []
                for a in range(len(tmp.the_weight)):
                    w.append(str(tmp.the_weight[a]))
                st =[]
                for a in range(len(tmp.stat_list)):    
                    st.append(str(round(tmp.stat_list[a],5)))
                tmp_list = [expect_reward,y,' '.join(tmp.the_names), ' '.join(w), tmp.train_reward, tmp.train_ann_reward,tmp.train_risk,tmp.train_mdd,tmp.test_reward,tmp.test_ann_reward,tmp.test_risk,tmp.test_mdd,tmp.success_ratio,tmp.success_stat,' '.join(st)]
                
                ans_list = [spy]
                ans_list.append([0,0,0,' '.join(tmp.the_names), ' '.join(w)])
                legends = ['SPY','our_comb']
                compare_reward_plot(filepath+'fig_US_r/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)

                
                ans_list.append(green_horn_comb)
                ans_list.append(mr_market_classic)
                
                legends = ['SPY','our_comb','green_horn','Mr.market_classic']
                compare_reward_plot(filepath+'fig_US_r_comp/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)
                compare_plot_3D(filepath+'fig_US_3D_comp/',ans_list,legends,y,train_nnnn,test_nnnn,expect_reward,mode,use)

                SPY_compare.append(tmp_list)
                big_list.append(tmp_list)

            else:
                SPY_compare.append([])
                big_list.append([])
            # break
        if len(SPY_compare[0])>0:
            df = pd.DataFrame(SPY_compare,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
            df.to_csv(filepath+'US_compare_'+str(expect_reward)+'.csv',index=False)
        # break

    df_big =  pd.DataFrame(big_list,columns=['expect_reward','year','names','weights','train_reward','train_ann_reward','train_risk','train_mdd','test_reward','test_ann_reward','test_risk','test_mdd','success_ratio','success_stat','stat_list']) #,'aETF_ann_r','aETF_risk','aETF_mdd'
    df_big.to_csv(filepath+'US_compare_all.csv',index=False)



# %%
if __name__ == '__main__':
    ## 輸出路徑
    filepath = 'D:/Alia/Documents/asset allocation/experiance/'
    os.mkdir(filepath)

    ### test時間長度
    test_nnnn = 3

    ########### 圖的參數 ############
    ### y1&y2=要跑的年份範圍    
    y1 = 2018
    y2 = 2019
    ### ab=要跑的expect_reward
    # ab=[3,4,5,6,7,8,9] 
    ab=[5] 
    # ab=[5,6,9,10,12,13]

    ########## 大表的參數 ############
    ### yyyys=大表要跑的年份
    # yyyys=[2012,2014,2016] 
    yyyys=[2016]
    

    # run_once(2015,3,5,4,1,0.05)

    ### 選股+跑大表(畫skyline)
    # os.mkdir(filepath+'big_table/')
    # run_big_table(filepath,test_nnnn=test_nnnn,yyyys=yyyys)
 
    ############(美國)畫圖#################
    # os.mkdir(filepath+'fig_US_sim/')
    # os.mkdir(filepath+'fig_US_r/')
    # os.mkdir(filepath+'fig_US_sim_comp/')
    # os.mkdir(filepath+'fig_US_r_comp/') 
    # os.mkdir(filepath+'fig_US_3D_comp/')
    ### 跟大盤比較的金流模擬圖+報酬率圖+選出來的組合輸出成csv 
    # run_plot_US(filepath,y1=2018,y2=y2,ab=[5,6,9,10,12,13],test_nnnn=test_nnnn)
    ### 跟大盤+已知組合比較的金流模擬圖+報酬率圖+選出來的組合輸出成csv 
    # run_plot_US_compare(filepath,y1=2018,y2=y2,ab=[5,6,9,10,12,13],test_nnnn=test_nnnn)
    
    #############(台灣)畫圖###################
    os.mkdir(filepath+'fig_TW_sim/')
    os.mkdir(filepath+'fig_TW_r/')
    os.mkdir(filepath+'fig_TW_sim_comp/')
    os.mkdir(filepath+'fig_TW_r_comp/')
    os.mkdir(filepath+'fig_TW_3D_comp/')
    ### 跟大盤比較的金流模擬圖+報酬率圖+選出來的組合輸出成csv 
    # run_plot_TW(filepath,y1=y1,y2=y2,ab=ab,test_nnnn=test_nnnn)
    ### 跟大盤+元大比較的金流模擬圖各一+報酬率圖各一+選出來的組合輸出成csv 
    run_plot_TW_compare(filepath,y1=y1,y2=y2,ab=ab,test_nnnn=test_nnnn)

    ### 跑單股的大表(畫skyline)(也可改成已知組合)
    # filepath_SPY = 'D:/Alia/Documents/109-1/資產配置/experience_new/0050_test/'
    # os.mkdir(filepath_SPY)
    # run_SPY(filepath_SPY,filename='SPY_5y',etf_name='SPY',ans_weight='1.0',test_nnnn=5,y1=2012,y2=2017)                                     
    # run_SPY(filepath_SPY,filename='grow',etf_name='00646.TW 0050.TW 006206.TW 00661.TW 00660.TW 00697B.TW 00720B.TW 00635U.TW 00682U.TW',ans_weight='0.1791 0.2301 0.1082 0.0747 0.0458 0.1150 0.0271 0.0782 0.1418',test_nnnn=test_nnnn,y1=2018,y2=2019)                                     
    
