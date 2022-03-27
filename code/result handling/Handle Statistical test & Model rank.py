#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[91]:


series_list = ["Nori", "Sori"]
H_list = ["H"+str(h) for h in range(2,11)]
path = "C:/D/HUST/research_flu_forecast/experiment/result/New/"
save_path = path + "Model Compare/csv_files/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

dual_ind_list = [H_list, ["Rank", "Sig"]]
ind = pd.MultiIndex.from_product(dual_ind_list, names=["Horizon", "item"])
col = ["Rank"+str(r+1) for r in range(6)]


# ## Statistical Metrics

# In[92]:


metrics_list = ["MAPE","RMSE"]

for series in series_list:
    metric_file = pd.read_excel(path + "Statistical Metrics/total_statistical_metrics_averaged.xlsx",sheet_name = series,index_col = [0,1])
    
    for metric in metrics_list:
        mr_l = []
        for h_n,H in enumerate(H_list):
            ms = metric_file.loc[metric, H].sort_values()
            ms_l = ms.index.tolist()
            
            st_file = pd.read_excel(path + "Statistical test/" + series + "_total_" + metric + "_stat_Nemenyi.xlsx",sheet_name = H,index_col = 0)
            st_l = []
            for m_n,model in enumerate(ms_l):    
                sig_str = ""
                for model_comp in ms_l[0:m_n]:
                    if (model_comp == 'SVR_Iter') or (model == 'MLP_MIMO') or (np.isnan(st_file[model][model_comp])):
                        pvalue = st_file[model_comp][model]
                    else:
                        pvalue = st_file[model][model_comp]

                    if pvalue < 0.05:
                        sig_str += ("*>" + model_comp + '\n')
                        
                st_l.append(sig_str[:-1])
            
            mr_l += [ms_l,st_l]
        mr_df = pd.DataFrame(mr_l, index= ind, columns= col)
        mr_df.to_csv(save_path + series + "_" + metric + "_" + "model_compare.csv")


# ## Outbreak Metrics

# In[94]:


metrics_list = ["PWD","Outbreak_MAE"]

for series in series_list:
    
    for metric in metrics_list:
        metric_file = pd.read_excel(path + metric + "/" + metric + "_total_ave.xlsx",sheet_name = series,index_col = [0,1])
        mr_l = []
        
        for h_n,H in enumerate(H_list):
            ms = metric_file.loc["outbreak_ave", H].sort_values()
            ms_l = ms.index.tolist()
            
            st_file = pd.read_excel(path + "Statistical test/" + series + "_total_" + metric + "_stat_Nemenyi.xlsx",sheet_name = H,index_col = 0)
            st_l = []
            for m_n,model in enumerate(ms_l):    
                sig_str = ""
                for model_comp in ms_l[0:m_n]:
                    if (model_comp == 'SVR_Iter') or (model == 'MLP_MIMO') or (np.isnan(st_file[model][model_comp])):
                        pvalue = st_file[model_comp][model]
                    else:
                        pvalue = st_file[model][model_comp]

                    if pvalue < 0.05:
                        sig_str += ("*>" + model_comp + '\n')
                
                st_l.append(sig_str[:-1])
            
            mr_l += [ms_l,st_l]
        mr_df = pd.DataFrame(mr_l, index = ind, columns = col)
        mr_df.to_csv(save_path + series + "_" + metric + "_" + "model_compare.csv")


# ## try

# In[69]:


np.isnan(st_file['MLP_Dir']['SVR_Dir'])  #正确的判断numpy数组中空值的方法


# In[13]:


a = "*>" + "SVR_Iter" + '\n' + "*>" +"MLP_Dir"   #'\n'为换行符，经测试，输出到.csv时可以呈现换行格式，但输出为excel时不行。
m = pd.Series(a)
m.to_csv("./try.csv")

