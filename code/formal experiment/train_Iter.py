#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import TimeSeriesSplit
#from sklearn.metrics import mean_squared_error


# In[ ]:


def import_trainXy(train_file):
    X = pd.read_excel(train_file,sheet_name="X")
    y = pd.read_excel(train_file,sheet_name="y")
    X_train = np.array(X)
    y_train = np.array(y)
    return X_train, y_train


# In[ ]:


#训练模型，k折交叉验证+网格搜索优化参数
def training_model(train_data_file, ML_model, feature_mask, params_dict, F, H, step = None):                #p_i:PSO中的一个粒子; F:feature dimension
    X_train, y_train_multi = import_trainXy(train_data_file)
    y_train = y_train_multi[:,0]
    
    #feature 
#    feature_mask = p_i[-F:]
    feature_index = np.arange(F)
    selected_feature_index = feature_index[feature_mask == 1]
    selected_X_train = X_train[0:X_train.shape[0], feature_mask == 1]  

    
    #feature_scale: log transformation
    X_train_scaled = np.log10(selected_X_train)
    y_train_scaled = np.log10(y_train)
    
    
    #model and parameters
#    model = SVR(kernel='rbf', gamma = param_gamma, C = param_C, epsilon = param_eps)    
    model = ML_model(**params_dict)                #python中传入（关键字）参数组合的方式：**    
    
    #time series spilt cv
#    tscv = TimeSeriesSplit(n_splits = 5)
#    cv_MSE = np.zeros(5)
#    i = 0
#    for train_index, test_index in tscv.split(X_train_scaled):
#        cv_X_train, cv_X_test = X_train_scaled[train_index], X_train_scaled[test_index]
#        cv_y_train, cv_y_test = y_train_scaled[train_index], y_train_scaled[test_index]
#    
#        model.fit(cv_X_train,cv_y_train.ravel())
#        cv_y_pred = model.predict(cv_X_test)
#        cv_MSE[i] = mean_squared_error(cv_y_test, cv_y_pred)
#        i += 1
#        
#        '''
#        pipe_SVR.fit(cv_X_train,cv_y_train.ravel())
#        cv_y_pred = pipe_SVR.predict(cv_X_test)
#        cv_MSE[i] = mean_squared_error(cv_y_test, cv_y_pred)
#        i += 1
#        '''
#    train_MSE = cv_MSE.mean()   
    
    
    # normal k-fold cv
    cv_results = cross_validate(estimator = model, X = X_train_scaled, y = y_train_scaled, scoring = "neg_mean_squared_error", cv = 5 ,n_jobs = -1)
    train_MSE = cv_results['test_score'].mean()
    train_MSE = abs(train_MSE)
    
    
    model.fit(X_train_scaled, y_train_scaled.ravel())
    
#    y_train_pred = model.predict(X_train_scaled)
#    y_train_pred = y_train_pred_scaled * (max_value - min_value) + min_value
#    
#    train_MSE = mean_squared_error(y_train_scaled, y_train_pred)
    
    return train_MSE, model, selected_feature_index

