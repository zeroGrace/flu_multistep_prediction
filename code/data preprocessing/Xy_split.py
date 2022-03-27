#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[2]:


#201001-202012 num of training set
total_len = 533
cut_21 = int(total_len * (2/3))
cut_82 = int(total_len * 0.8)
print("train2test1_num of training point:",cut_21)
print("train8test2_num of training point:",cut_82)


# In[18]:


def split_xy(series, path, label):
    total_len = len(series)
       
    for n in range(1,13):
        for m in range(16,21):
            setX = []
            setY = []
            cut_point = int(total_len * (2/3)) - (m + n) + 1           #train:test = 2:1
            #cut_point = int(total_len * 0.8) - (m + n) + 1            #train:test = 8:2
            
            for i in range(len(series)-(n+m)+1): 
                setX.append(series[i:i+m])
                setY.append(series[i+m:i+n+m])
            
            X_train = pd.DataFrame(setX[:cut_point])
            y_train = pd.DataFrame(setY[:cut_point])
            
            X_test = pd.DataFrame(setX[cut_point:])
            y_test = pd.DataFrame(setY[cut_point:])
            
            train_filename = path + "train_" +  label + "_X" + str(m) + "_y" + str(n) + ".xlsx"            
            train_writer = pd.ExcelWriter(train_filename)
            X_train.to_excel(train_writer,'X',index=False)
            y_train.to_excel(train_writer,'y',index=False)
            train_writer.save()
            
            test_filename = path  + "test_" + label +  "_X" + str(m) + "_y" + str(n) + ".xlsx"
            test_writer = pd.ExcelWriter(test_filename)
            X_test.to_excel(test_writer,'X',index=False)
            y_test.to_excel(test_writer,'y',index=False)
            test_writer.save()


# In[19]:


data_list = ["201001-202012_train2test1", "201001-202012_train8test2"]
data_label = data_list[1]
type_keys = ["N_ILI_ori", "S_ILI_ori", "N_ILI_sadj", "S_ILI_sadj", "N_ILI_s", "S_ILI_s", "N_ILI_log", "S_ILI_log"]
data_dict = {}
saving_path_dict = {}

data_file = pd.read_excel("C:\\D\\HUST\\research_flu_forecast\\data for coding\\" + data_label + "\\ILI_all.xlsx")
colname_list = data_file.columns.tolist()        #列名（列索引）的集合
 

for num,k in enumerate(type_keys):
    data_dict[k] = data_file[colname_list[num+1]].tolist()
    saving_path_dict[k] = "C:\\D\\HUST\\research_flu_forecast\\data for coding\\" + data_label + "\\Xy\\" + k + "\\"
    
    split_xy(data_dict[k], saving_path_dict[k], k)



