#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


series_list = ["Nori", "Sori"]
model_list = ['SVR_Iter', 'SVR_Dir', 'SVR_MIMO', 'MLP_Iter', 'MLP_Dir','MLP_MIMO']

open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/"
save_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Statistical Metrics/"

Model_Num = 6
RANKS = 20


# In[3]:


#total_Metric_gather(for statistic analysis): 2 series × 2 metric
save_folder1 = "total_metric_gather"  
if os.path.exists(save_path + save_folder1) == False:
    os.makedirs(save_path + save_folder1)

for series_name in series_list:
    for metric_name in ['MAPE','RMSE']:
        wr_1 = pd.ExcelWriter(save_path + save_folder1 + "/" + series_name + "_total_" + metric_name + "_gather.xlsx")
        for H in range(2,11):
            index_rank = ['rank' + str(i+1) for i in range(RANKS)]
            column_model = model_list            
            gather_table = np.zeros((RANKS,Model_Num))
            
            for num, model_name in enumerate(model_list):
                metric_file = open_path + series_name + "/" + model_name + "/aggregate_metric/Total_Metrics_" + series_name + "_"  + model_name + ".xlsx"
                metric_table = pd.read_excel(metric_file, sheet_name = "H"+str(H))
                metric = metric_table[metric_name]
                gather_table[:,num] = np.array(metric)
                
            df1 = pd.DataFrame(gather_table,index = index_rank ,columns = column_model)
            df1.to_excel(wr_1,'H'+str(H))
        wr_1.save()


# In[4]:


#step_Metric_gather(for statistic analysis): 2 series × 2 metric × 9 horizons
save_folder2 = "step_metric_gather"
if os.path.exists(save_path + save_folder2) == False:
    os.makedirs(save_path + save_folder2)
    
for series_name in series_list:
    for metric_name in ['MAPE','RMSE']:
        for H in range(2,11):
            wr_2 = pd.ExcelWriter(save_path + save_folder2 + "/" + series_name + "_step_" + metric_name + "_H" + str(H) +"_gather.xlsx")
            
            for step in range(1, H+1):
                index_rank = ['rank' + str(i+1) for i in range(RANKS)]
                column_model = model_list            
                gather_table = np.zeros((RANKS,Model_Num))
            
                for num, model_name in enumerate(model_list):
                    metric_file = open_path + series_name + "/" + model_name + "/aggregate_metric/" + metric_name + "_Step_" + series_name + "_"  + model_name + ".xlsx"
                    metric_table = pd.read_excel(metric_file, sheet_name = "H" + str(H))
                    metric = metric_table["step"+str(step)]
                    gather_table[:,num] = np.array(metric)
                
                df2 = pd.DataFrame(gather_table,index = index_rank ,columns = column_model)
                df2.to_excel(wr_2,"step" + str(step))
            wr_2.save()


# In[12]:


#total_Metrics_averaged
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Statistical Metrics/total_metric_gather/"

#多级索引
dual_ind_list = [['MAPE','RMSE'],model_list]
metric_ind = pd.MultiIndex.from_product(dual_ind_list, names = ['Metric','Model'])
col= ["H"+str(h) for h in range(2,11)]
    
wr_3 = pd.ExcelWriter(save_path + "/total_statistical_metrics_averaged.xlsx")
for series_name in series_list:
    gather_table = np.zeros((Model_Num*2,9))    
    for n,metric_name in enumerate(['MAPE','RMSE']):
        for H in range(2,11):
            metric_file = open_path + series_name + "_total_"  + metric_name  + "_gather" + ".xlsx"
            metric_table = pd.read_excel(metric_file, sheet_name = "H" + str(H),index_col = 0)
            metric = np.array(metric_table).T
            gather_table[(n*6):(n+1)*6,H-2] = np.mean(metric,axis = 1)
            
    df3 = pd.DataFrame(gather_table,index = metric_ind ,columns = col)
    df3.to_excel(wr_3,series_name)
wr_3.save()


# In[16]:


#step_Metrics_averaged
open_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Statistical Metrics/step_metric_gather/"

#多级索引
dual_ind_list = [['MAPE','RMSE'],model_list]
metric_ind = pd.MultiIndex.from_product(dual_ind_list, names = ['Metric','Model'])
col_list = []
for H in range(2,11):
    for h in range(1,H+1):
        col_list.append(['H'+str(H),'step'+str(h)])
col_df = pd.DataFrame(col_list)
col = pd.MultiIndex.from_frame(col_df,names = ['Horizon','Step'])    

wr_4 = pd.ExcelWriter(save_path + "/step_statistical_metrics_averaged.xlsx")
for series_name in series_list:
    gather_table = np.zeros((Model_Num*2,54))
    
    for n,metric_name in enumerate(['MAPE','RMSE']):    
        
        H_position = 0
        for H in range(2,11):
            for step in range(H):    
                metric_file = open_path + series_name + "_step_"  + metric_name  + "_H" + str(H) + "_gather" + ".xlsx"
                metric_table = pd.read_excel(metric_file, sheet_name = "step" + str(step+1),index_col=0)
                metric = np.array(metric_table).T
                gather_table[(n*6):(n+1)*6, H_position + step] = np.mean(metric,axis = 1)
            H_position += H
            
    df4 = pd.DataFrame(gather_table,index = metric_ind ,columns = col)
    df4.to_excel(wr_4, series_name)
wr_4.save()


# In[3]:


#-------------------test-------------------------


# In[5]:


a = np.array([[2,3,4],[3,4,5]])
a


# In[6]:


np.mean(a, axis=1)


# In[10]:


metric_table = pd.read_excel(metric_file, sheet_name = "H" + str(H),index_col=0)
metric_table

