#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[ ]:


def import_testXy(test_file, H):
    X = pd.read_excel(test_file,sheet_name="X")
    y = pd.read_excel(test_file,sheet_name="y")
    
    X_test = np.array(X)
    y_test = np.array(y)    
    
    y_series = np.concatenate((y_test[:,0], y_test[-1,1:H]), axis = 0)
    y_series_length = len(y_series)
    
    return X_test, y_test, y_series, y_series_length

#预测结果合成整条序列
def data_series(y_test_pred, y_series_length, H):
    temp_matrix = np.zeros((len(y_test_pred),y_series_length))
    
    for i in range(len(y_test_pred)):
        temp_matrix[i,i:i+H] = y_test_pred[i]
    
    sum_vector = np.sum(temp_matrix, axis = 0)
    
    divisor = np.full((y_series_length, ),H)
    for i in range(1,H):
        divisor[i-1] = i
        divisor[-i] = i
        
    y_test_pred_series = sum_vector / divisor
        
    return y_test_pred_series
    
def test_model(test_file, model_set, F, H, series_label, sea_test_file, feature_index_set):  
    
    X_test, y_test, y_test_series, y_series_length = import_testXy(test_file, H)
    y_test_pred_scaled = np.zeros((X_test.shape[0], H))
  
    
    #predict y
    for step, model in enumerate(model_set):
        selected_X_test = X_test[:, feature_index_set[step]]
        
        #scale X_test:log transformation    
        X_test_scaled = np.log10(selected_X_test)        
        
        y_test_pred_col = model.predict(X_test_scaled)
        y_test_pred_scaled[:, step] = y_test_pred_col
           
    
    #rescale the y_test_pred  
    y_test_pred = np.power(10, y_test_pred_scaled)
    
    
    #print("y_test_pred:")
    #print(y_test_pred) 
    #print("y_test_pred.shape:", y_test_pred.shape)   
    
    if series_label in ["N_ILI_sadj","S_ILI_sadj"]:
        sea = pd.read_excel(sea_test_file, sheet_name="y")
        sea_np = np.array(sea)
        
        y_test = y_test + sea_np
        y_test_pred = y_test_pred + sea_np
        y_test_series = y_test_series + np.concatenate((sea_np[:,0],sea_np[-1,1:H]), axis = 0)
    
    y_test_pred_series = data_series(y_test_pred, y_series_length, H)
    
    return y_test, y_test_pred, y_test_series, y_test_pred_series 


# In[ ]:


#作图（测试集预测结果）
def graph(y_test, y_test_pred, H, F):    
    date = list(range(len(y_test)))
#    plt.title("H = "+str(H)+";F = "+str(F))
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.ylim(0,10)
    plt.plot(date, y_test, c='blue')
    plt.plot(date, y_test_pred, c='red')
    plt.show()


# In[ ]:


#评价
def evaluate(y_test, y_test_pred, H):    
    
    mape_point = np.abs((y_test - y_test_pred)/y_test)
    se_point = np.power((y_test - y_test_pred),2)               #squared error
    
    #series average
#    series_length = len(y_test)+ H - 1
#    mape_series = data_series(mape_point, series_length, H)
#    mape = np.mean(mape_series)   
#    
#    rmse_series = data_series(rmse_point, series_length, H)
#    rmse = np.sqrt(np.mean(rmse_series))
    
    #total performance
    mape = np.mean(mape_point)
    rmse = np.sqrt(np.mean(se_point))
    
    #preformance of each step
    mape_step = np.mean(mape_point,axis = 0)
    mape_step_std = np.std(mape_point,axis = 0)
    
    rmse_step = np.sqrt(np.mean(se_point, axis = 0))
    rmse_step_std = np.std(np.sqrt(se_point), axis = 0)      #rmse的std：相当于是每一step下time point error绝对值的标准差
      
#    print("total_MAPE:", mape)
#    print("total_RMSE:", rmse)
    
    return mape, rmse, mape_step, mape_step_std, rmse_step, rmse_step_std

