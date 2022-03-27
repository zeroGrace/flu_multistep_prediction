#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[3]:


#cut test part of ILI series
ili_file = pd.read_excel('C:/D/HUST/research_flu_forecast/data for coding/ILI_all.xlsx')
test_ILI = {}
test_ILI['Nori'] = ili_file['n_ili'].values[-187:]
test_ILI['Sori'] = ili_file['s_ili'].values[-187:]
test_weektag = ili_file['weektag'].values[-187:]


# In[2]:


series_list = ["Nori", "Sori"]
model_list = ['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']
R = 20


# ## Outbreak RMSE

# ### Calculate each Outbreak RMSE

# In[ ]:


#calculate each Outbreak RMSE
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/"
ob_rmse_save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"

for series_name in series_list:
    test_series = test_ILI[series_name]
    if os.path.exists(ob_rmse_save_path + "/" + series_name) == False:
        os.makedirs(ob_rmse_save_path + "/" + series_name)
    
    for model_name in model_list:
        for H in range(2,11):
            ob_rmse_file_name = "Outbreak_RMSE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
            wr = pd.ExcelWriter(ob_rmse_save_path + "/" + series_name + "/" + ob_rmse_file_name)
            col = ['outbreak1','outbreak2','outbreak3','outbreak_mean']
            ind = ['step'+str(x+1) for x in range(H)]
            ind.append('step_mean')
            
            for r in range(R):
                file_name = series_name + "_" + model_name + "_y" +str(H) + "_rank" + str(r+1) + ".xlsx"
                result_file = pd.read_excel(open_path + series_name + "/" + model_name + "/"  + file_name, sheet_name = 'y_test_pred')
                true_len = len(test_series)
                pred_len = result_file.shape[0]
                
                #各step对齐到原序列的时间点
                t_m = np.zeros((H+1,true_len))
                t_m[H] = test_series
                step_list = list(range(H))
                step_list.reverse()                    #list.reverse()没有返回值，不返回新的反序list，只是对原list的元素进行反向排序
                for step in step_list:           
                    t_m[step][(step+1-H)-pred_len:true_len + (step+1-H)] = result_file['step'+ str(step+1)].values
                            
                #三个爆发期（不包括2016-2017）的具体时间段：每年11月初-次年2月（week45-次年week8）；64-79、116-131、168-183
                outbreaks = [t_m[:,63:79].copy(),t_m[:,115:131].copy(),t_m[:,167:183].copy()]
                ob_rmse = np.zeros((H+1,3+1))
                
                for n in range(3):
                    for step in range(H):
                         ob_rmse[step,n] = np.sqrt(np.mean((outbreaks[n][H]-outbreaks[n][step])**2))   

                #求均值
                ob_rmse[:,3] = np.mean(ob_rmse[:,:3],axis=1)
                ob_rmse[H,:] = np.mean(ob_rmse[:H,:],axis=0)
                
                ob_rmse_df = pd.DataFrame(ob_rmse,index = ind, columns = col)
                ob_rmse_df.to_excel(wr,'rank'+str(r+1))
            wr.save()


# ### Outbreak RMSE averaged

# In[6]:


#Outbreak RMSE averaged
#1)table(repeated experiment) averaged 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/repeat average/"
for series_name in series_list:
    for model_name in model_list:
        save_name = "repAve_Outbreak_RMSE_" + series_name + "_" + model_name + ".xlsx"
        wr1 = pd.ExcelWriter(save_path + save_name)
        for H in range(2,11):
            col = ['outbreak1','outbreak2','outbreak3','outbreak_mean']
            ind = ['step'+str(x+1) for x in range(H)]
            ind.append('step_mean')
            
            file_name = "Outbreak_RMSE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
            add_table = 0.0
            for r in range(R):
                #最新版pd.read.values读取内容时，默认存在列索引而不存在行索引，因此若存在行索引，则会被包括在结果中，需剔除
                add_table += pd.read_excel(open_path+series_name+ "/" + file_name,sheet_name = 'rank'+str(r+1)).values[:,1:]  
            mean_table = add_table/R
            mean_df = pd.DataFrame(mean_table, index = ind, columns = col)
            mean_df.to_excel(wr1,'H'+str(H))
        wr1.save()


# #### Outbreak_RMSE total averaged 

# In[7]:


#Outbreak RMSE averaged
#2)total&outbreak averaged gather;first model then Outbreak_RMSE_type
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"

#pandas 多级索引 
dual_ind_list = [['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO'],['outbreak1','outbreak2','outbreak3','outbreak_ave']]
ob_rmse_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Model','Outbreak_RMSE_type'])
col= ["H"+str(h) for h in range(2,11)]

wr2 = pd.ExcelWriter(save_path + "Outbreak_RMSE_total_ave_modelFirstCol.xlsx")
for series_name in series_list:
    ob_rmse_table = np.zeros((4*6,9))
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_RMSE_" + series_name + "_" + model_name + ".xlsx"
        for H in range(2,11):
            ob_rmse_table[num_m*4:(num_m+1)*4, H-2] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[H,1:]
    ob_rmse_df = pd.DataFrame(ob_rmse_table,index = ob_rmse_ind,columns=col)
    ob_rmse_df.to_excel(wr2,series_name)
wr2.save()


# In[8]:


#Outbreak RMSE averaged
#2)total&outbreak averaged gather;first Outbreak_RMSE_type then model 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"

#pandas 多级索引 
dual_ind_list = [['outbreak1','outbreak2','outbreak3','outbreak_ave'],['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']]
ob_rmse_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Outbreak_RMSE_type','Model'])
col= ["H"+str(h) for h in range(2,11)]

wr2 = pd.ExcelWriter(save_path + "Outbreak_RMSE_total_ave.xlsx")
for series_name in series_list:
    ob_rmse_table = np.zeros((4*6,9))
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_RMSE_" + series_name + "_" + model_name + ".xlsx"
        for H in range(2,11):
            for ob_n in range(0,4):
                ob_rmse_table[num_m+(ob_n*6),H-2] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[H,ob_n+1]

    ob_rmse_df = pd.DataFrame(ob_rmse_table,index = ob_rmse_ind,columns=col)
    ob_rmse_df.to_excel(wr2,series_name)
wr2.save()


# #### Outbreak_RMSE step averaged

# In[9]:


#Outbreak step gather and averaged: first Outbreak_RMSE_type then model 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"

#pandas 多级索引 
dual_ind_list = [['outbreak1','outbreak2','outbreak3','outbreak_ave'],['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']]
ob_rmse_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Outbreak_RMSE_type','Model'])

col_list = []
for H in range(2,11):
    for h in range(1,H+1):
        col_list.append(['H'+str(H),'step'+str(h)])
col_df = pd.DataFrame(col_list)
col = pd.MultiIndex.from_frame(col_df,names = ['Horizon','Step'])

wr3 = pd.ExcelWriter(save_path + "Outbreak_RMSE_step_ave.xlsx")
for series_name in series_list:
    ob_rmse_table = np.zeros((4*6,54))     #54：2+3+4+...+10
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_RMSE_" + series_name + "_" + model_name + ".xlsx"
        
        H_position = 0
        for H in range(2,11):
            for ob_n in range(0,4):
                ob_rmse_table[num_m+(ob_n*6),H_position:(H_position+H)] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[0:H,ob_n+1]                
            H_position += H
            
    ob_rmse_df = pd.DataFrame(ob_rmse_table,index = ob_rmse_ind,columns=col)
    ob_rmse_df.to_excel(wr3,series_name)
wr3.save()


# ### Outbreak RMSE gather

# In[10]:


#total and outbreak gather
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_RMSE/Outbreak_RMSE_gather/"
ob_sheet = ['Outbreak1','Outbreak2','Outbreak3','total']

col = model_list
ind = ['rank' + str(i+1) for i in range(R)]

for series_name in series_list:
    for H in range(2,11):
        save_name = "Outbreak_RMSE_total_gather_" + series_name + "_H" +str(H) + ".xlsx"
        wr1 = pd.ExcelWriter(save_path + save_name)
        
        for ob_n in range(0,4):
            ob_rmse_table = np.zeros((20,6))
            
            for m_n, model_name in enumerate(model_list):
                open_name = "Outbreak_RMSE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
                
                for r in range(R):
                    ob_rmse_table[r, m_n] = pd.read_excel(open_path + series_name+ "/" + open_name, sheet_name = 'rank'+str(r+1)).values[H, ob_n+1]
        
            ob_rmse_df = pd.DataFrame(ob_rmse_table, index = ind, columns = col)
            ob_rmse_df.to_excel(wr1,ob_sheet[ob_n])
        wr1.save()


# ## Outbreak MAE

# ### Calculate each Outbreak MAE

# In[11]:


#calculate each Outbreak MAE
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/"
ob_mae_save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"

for series_name in series_list:
    test_series = test_ILI[series_name]
    if os.path.exists(ob_mae_save_path + "/" + series_name) == False:
        os.makedirs(ob_mae_save_path + "/" + series_name)
    
    for model_name in model_list:
        for H in range(2,11):
            ob_mae_file_name = "Outbreak_MAE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
            wr = pd.ExcelWriter(ob_mae_save_path + "/" + series_name + "/" + ob_mae_file_name)
            col = ['outbreak1','outbreak2','outbreak3','outbreak_mean']
            ind = ['step'+str(x+1) for x in range(H)]
            ind.append('step_mean')
            
            for r in range(R):
                file_name = series_name + "_" + model_name + "_y" +str(H) + "_rank" + str(r+1) + ".xlsx"
                result_file = pd.read_excel(open_path + series_name + "/" + model_name + "/"  + file_name, sheet_name = 'y_test_pred')
                true_len = len(test_series)
                pred_len = result_file.shape[0]
                
                #各step对齐到原序列的时间点
                t_m = np.zeros((H+1,true_len))
                t_m[H] = test_series
                step_list = list(range(H))
                step_list.reverse()                    #list.reverse()没有返回值，不返回新的反序list，只是对原list的元素进行反向排序
                for step in step_list:           
                    t_m[step][(step+1-H)-pred_len:true_len + (step+1-H)] = result_file['step'+ str(step+1)].values
                            
                #三个爆发期（不包括2016-2017）的具体时间段：每年11月初-次年2月（week45-次年week8）；64-79、116-131、168-183
                outbreaks = [t_m[:,63:79].copy(),t_m[:,115:131].copy(),t_m[:,167:183].copy()]
                ob_mae = np.zeros((H+1,3+1))
                
                for n in range(3):
                    for step in range(H):
                         ob_mae[step,n] = np.mean(abs(outbreaks[n][H]-outbreaks[n][step]))   

                #求均值
                ob_mae[:,3] = np.mean(ob_mae[:,:3],axis=1)
                ob_mae[H,:] = np.mean(ob_mae[:H,:],axis=0)
                
                ob_mae_df = pd.DataFrame(ob_mae,index = ind, columns = col)
                ob_mae_df.to_excel(wr,'rank'+str(r+1))
            wr.save()


# ### Outbreak MAE averaged

# In[12]:


#Outbreak averaged
#1)table(repeated experiment) averaged 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/repeat average/"
for series_name in series_list:
    for model_name in model_list:
        save_name = "repAve_Outbreak_MAE_" + series_name + "_" + model_name + ".xlsx"
        wr1 = pd.ExcelWriter(save_path + save_name)
        for H in range(2,11):
            col = ['outbreak1','outbreak2','outbreak3','outbreak_mean']
            ind = ['step'+str(x+1) for x in range(H)]
            ind.append('step_mean')
            
            file_name = "Outbreak_MAE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
            add_table = 0.0
            for r in range(R):
                add_table += pd.read_excel(open_path+series_name+ "/" + file_name,sheet_name = 'rank'+str(r+1)).values[:,1:]
            mean_table = add_table/R
            mean_df = pd.DataFrame(mean_table, index = ind, columns = col)
            mean_df.to_excel(wr1,'H'+str(H))
        wr1.save()


# #### Outbreak_MAE total averaged 

# In[14]:


#Outbreak gather and averaged
#2)total&outbreak averaged gather;first model then Outbreak_MAE_type
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"

#pandas 多级索引 
dual_ind_list = [['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO'],['outbreak1','outbreak2','outbreak3','outbreak_ave']]
ob_mae_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Model','Outbreak_MAE_type'])
col= ["H"+str(h) for h in range(2,11)]

wr2 = pd.ExcelWriter(save_path + "Outbreak_MAE_total_ave_modelFirstCol.xlsx")
for series_name in series_list:
    ob_mae_table = np.zeros((4*6,9))
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_MAE_" + series_name + "_" + model_name + ".xlsx"
        for H in range(2,11):
            ob_mae_table[num_m*4:(num_m+1)*4, H-2] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[H,1:]
    ob_mae_df = pd.DataFrame(ob_mae_table,index = ob_mae_ind,columns=col)
    ob_mae_df.to_excel(wr2,series_name)
wr2.save()


# In[15]:


#Outbreak_MAE gather and averaged
#2)total&outbreak averaged gather;first Outbreak_MAE_type then model 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"

#pandas 多级索引 
dual_ind_list = [['outbreak1','outbreak2','outbreak3','outbreak_ave'],['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']]
ob_mae_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Outbreak_MAE_type','Model'])
col= ["H"+str(h) for h in range(2,11)]

wr2 = pd.ExcelWriter(save_path + "Outbreak_MAE_total_ave.xlsx")
for series_name in series_list:
    ob_mae_table = np.zeros((4*6,9))
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_MAE_" + series_name + "_" + model_name + ".xlsx"
        for H in range(2,11):
            for ob_n in range(0,4):
                ob_mae_table[num_m+(ob_n*6),H-2] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[H,ob_n+1]

    ob_mae_df = pd.DataFrame(ob_mae_table,index = ob_mae_ind,columns=col)
    ob_mae_df.to_excel(wr2,series_name)
wr2.save()


# #### Outbreak_MAE step averaged 

# In[16]:


#Outbreak step gather and averaged: first Outbreak_MAE_type then model 
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/repeat average/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"

#pandas 多级索引 
dual_ind_list = [['outbreak1','outbreak2','outbreak3','outbreak_ave'],['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']]
ob_mae_ind = pd.MultiIndex.from_product(dual_ind_list,names = ['Outbreak_MAE_type','Model'])

col_list = []
for H in range(2,11):
    for h in range(1,H+1):
        col_list.append(['H'+str(H),'step'+str(h)])
col_df = pd.DataFrame(col_list)
col = pd.MultiIndex.from_frame(col_df,names = ['Horizon','Step'])

wr3 = pd.ExcelWriter(save_path + "Outbreak_MAE_step_ave.xlsx")
for series_name in series_list:
    ob_mae_table = np.zeros((4*6,54))     #54：2+3+4+...+10
    for num_m,model_name in enumerate(model_list):
        open_file = "repAve_Outbreak_MAE_" + series_name + "_" + model_name + ".xlsx"
        
        H_position = 0
        for H in range(2,11):
            for ob_n in range(0,4):
                ob_mae_table[num_m+(ob_n*6),H_position:(H_position+H)] = pd.read_excel(open_path + open_file, sheet_name='H'+str(H)).values[0:H,ob_n+1]                
            H_position += H
            
    ob_mae_df = pd.DataFrame(ob_mae_table,index = ob_mae_ind,columns=col)
    ob_mae_df.to_excel(wr3,series_name)
wr3.save()


# ### Outbreak MAE gather

# In[17]:


#total and outbreak gather
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/Outbreak_MAE_gather/"
ob_sheet = ['Outbreak1','Outbreak2','Outbreak3','total']

col = model_list
ind = ['rank' + str(i+1) for i in range(R)]

for series_name in series_list:
    for H in range(2,11):
        save_name = "Outbreak_MAE_total_gather_" + series_name + "_H" +str(H) + ".xlsx"
        wr1 = pd.ExcelWriter(save_path + save_name)
        
        for ob_n in range(0,4):
            ob_mae_table = np.zeros((20,6))
            
            for m_n, model_name in enumerate(model_list):
                open_name = "Outbreak_MAE_" + series_name + "_" + model_name + "_y" +str(H) + ".xlsx"
                
                for r in range(R):
                    ob_mae_table[r, m_n] = pd.read_excel(open_path + series_name+ "/" + open_name, sheet_name = 'rank'+str(r+1)).values[H, ob_n+1]
        
            ob_mae_df = pd.DataFrame(ob_mae_table, index = ind, columns = col)
            ob_mae_df.to_excel(wr1,ob_sheet[ob_n])
        wr1.save()


# In[5]:


# only total gather
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/Outbreak_MAE_gather/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Outbreak_MAE/total_metric_gather/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

for series_name in series_list:
    save_name = series_name + "_total_Outbreak_MAE_gather.xlsx"
    wr = pd.ExcelWriter(save_path +save_name)
    
    for H in range(2,11):
        open_name = "Outbreak_MAE_total_gather_" + series_name + "_H" +str(H) + ".xlsx"
        total_df = pd.read_excel(open_path + open_name,sheet_name='total', index_col=0)
        total_df.to_excel(wr,"H"+str(H))
    wr.save()


# ## try

# In[7]:


col_list = []
for H in range(2,11):
    for h in range(1,H+1):
        col_list.append(['H'+str(H),'step'+str(h)])
col_df = pd.DataFrame(col_list)
col = pd.MultiIndex.from_frame(col_df)
col


# In[12]:


R = 10
open_file = "repAve_PWD_Nori_SVR_MIMO.xlsx"
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/PWD/repeat average/"
series_name = 'Nori'
H = 4


# In[3]:


dual_ind_list = [['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO'],['PWD_total','PWD_outbreak1','PWD_outbreak2','PWD_outbreak3']]
pd.MultiIndex.from_product(dual_ind_list,names = ['model','PWD_type'])


# In[8]:


try_ind = [[1,2],['a','b']]
ind = pd.MultiIndex.from_product(try_ind,names = ['first','second'])
df1 = pd.DataFrame(np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]),index = ind)
df1.to_excel("C:/D/HUST/research_flu_forecast/experiment/result/PWD/try.xlsx")


# In[25]:


H = 4
series_name = 'Nori'
test_series = test_ILI[series_name]
model_name = 'SVR_Iter'
r = 1
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/"

file_name = series_name + "_" + model_name + "_y" +str(H) + "_rank" + str(r) + ".xlsx"
result_file = pd.read_excel(open_path + series_name + "/" + model_name + "/"  + file_name, sheet_name = 'y_test_pred')
true_len = len(test_series)
pred_len = result_file.shape[0]

t_m = np.zeros((H+1,true_len))
t_m[H] = test_series
step_list = list(range(H))
step_list.reverse()                    #list.reverse()没有返回值，不返回新的反序list，只是对原list的元素进行反向排序
for step in step_list:           
    t_m[step][(step+1-H)-pred_len:true_len + (step+1-H)] = result_file['step'+ str(step+1)].values
    
t_m


# In[40]:


#算Outbreak_MSE
outbreaks = [t_m[:,50:100].copy(),t_m[:,100:155].copy(),t_m[:,155:].copy()]
pwd = np.zeros((H+1,3+1))
for n in range(3):
    pw_index = np.argmax(outbreaks[n], axis=1)
    for step in range(H):
         pwd[step,n] = abs(pw_index[H]-pw_index[step])   

#求均值
'''            #与下方等效
pwd[:H,3] = np.mean(pwd[:H,:3],axis=1)
pwd[H,:3] = np.mean(pwd[:H,:3],axis=0)
pwd[H,3] = np.mean(pwd[:H,:3])
pwd
'''
pwd[:,3] = np.mean(pwd[:,:3],axis=1)
pwd[H,:] = np.mean(pwd[:H,:],axis=0)
pwd


# In[29]:


outbreaks = [t_m[:,63:79].copy(),t_m[:,115:131].copy(),t_m[:,167:183].copy()]
outbreaks[0]


# In[29]:


a = list(range(4))
a.reverse()
a[-3:4]

