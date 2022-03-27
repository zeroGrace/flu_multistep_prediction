#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


open_path = 'C:/D/HUST/research_flu_forecast/experiment/result/New/'


# In[4]:


models = ['SVR','MLP']
metrics = ['MAPE','RMSE','PWE','Outbreak MAE']


# In[5]:


#Nori
data = pd.read_excel(open_path + "4_metrics_averaged.xlsx",sheet_name = 'Nori')
#plt.figure(figsize=(24,12))
#fig, axs = plt.subplots(2, 4)

axs = plt.figure(figsize = (12,6)).subplots(2, 4)
for m, model in enumerate(models):
    for n, metric in enumerate(metrics):
        iter_r = data.iloc[m*3 + n*6][2:11]
        dir_r = data.iloc[m*3 + n*6 + 1][2:11]
        mimo_r = data.iloc[m*3 + n*6 + 2][2:11]
       
        
        x_axis = list(range(2,11))
        
        axs[m,n].set_title(model)
        axs[m,n].set_xlabel('horizon')
        axs[m,n].set_ylabel(metric)
        
        axs[m,n].plot(x_axis, iter_r, '--r',label = 'Iter')
        axs[m,n].plot(x_axis, dir_r, '-.c',label = 'Dir')
        axs[m,n].plot(x_axis, mimo_r,':k',label = 'MIMO')
        axs[m,n].legend(loc = 'best')
        

plt.tight_layout()
plt.savefig('C:/D/HUST/research_flu_forecast/paper/figures & pics/Nori-metrics.png')
plt.show()
#加横轴纵轴的title


# In[7]:


data


# In[5]:


#Sori
data = pd.read_excel(open_path + "4_metrics_averaged.xlsx",sheet_name = 'Sori')

axs = plt.figure(figsize = (12,6)).subplots(2, 4)
for m, model in enumerate(models):
    for n, metric in enumerate(metrics):
        iter_r = data.iloc[m*3 + n*6][2:11]
        dir_r = data.iloc[m*3 + n*6 + 1][2:11]
        mimo_r = data.iloc[m*3 + n*6 + 2][2:11]
       
        
        x_axis = list(range(2,11))
        
        axs[m,n].set_title(model)
        axs[m,n].set_xlabel('horizon')
        axs[m,n].set_ylabel(metric)
        
        axs[m,n].plot(x_axis, iter_r, '--r',label = 'Iter')
        axs[m,n].plot(x_axis, dir_r, '-.c',label = 'Dir')
        axs[m,n].plot(x_axis, mimo_r,':k',label = 'MIMO')
        axs[m,n].legend(loc = 'best')
        

plt.tight_layout()
plt.savefig('C:/D/HUST/research_flu_forecast/paper/figures & pics/Sori-metrics.png')
plt.show()

#加横轴纵轴的title

