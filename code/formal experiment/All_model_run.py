#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import time
import os
from operator import itemgetter

import sys
sys.path.append(".")   #设置自定义包的搜索路径
import Params_determining_CLPSO as cp

import train_Iter
import train_Dir
import train_MIMO

import test_Iter
import test_Dir
import test_MIMO


from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor 
#sys.path.append("C:/D/HUST/research_flu_forecast/experiment/code/MSVR_python")   
from MSVR import MSVR

# In[ ]:
F = 20          #feature dimension
TIMES = 20       #repeat times                                                   #change

# data_sourse_list = ["201001-202012_train2test1", "201001-202012_train8test2"]

series_list = ["N_ILI_ori", "N_ILI_sadj", "S_ILI_ori", "S_ILI_sadj"]
model_key = ['SVR', 'MLP', 'KNN']
model_dict = {'SVR': SVR, 'MLP':MLPRegressor, 'KNN':KNeighborsRegressor}
strategy_key = ['Iter', 'Dir', 'MIMO']
train_dict = {'Iter': train_Iter, 'Dir': train_Dir, 'MIMO':train_MIMO}
test_dict = {'Iter':test_Iter, 'Dir':test_Dir, 'MIMO':test_MIMO}

save_path = "../result_new/"                                                   #change

data_sourse_label = "201001-202012_train2test1"
print(data_sourse_label)

total_start_time = time.time()
for series_label in series_list[2:3]:                                           #change:Sori
    type = series_label[:5]
    
    series_save_name = series_label[0] + series_label[-3:]
    if os.path.exists(save_path + series_save_name) == False:
        os.makedirs(save_path + series_save_name)
        
    for model_name in model_key[1:2]:                                            #change:MLP
        ML_model = model_dict[model_name]
        
        for strategy_name in strategy_key[1:2]:                                   #change:Dir
            if model_name == 'SVR' and strategy_name == 'MIMO':               
                ML_model = MSVR
            
            train = train_dict[strategy_name]
            test = test_dict[strategy_name]
                        
            model_folder_path = save_path + series_save_name + "/" 
            model_folder_name = model_name + "_"  + strategy_name
            if os.path.exists(model_folder_path + model_folder_name) == False:
                os.makedirs(model_folder_path + model_folder_name)
            
            file_path = model_folder_path + model_folder_name + "/"
            if os.path.exists(file_path + "aggregate_metric") == False:
                os.makedirs(file_path + "aggregate_metric")
            if os.path.exists(file_path + "TOP10") == False:
                os.makedirs(file_path + "TOP10")
            
            total_metric_name = "Total_Metrics_" + series_save_name + "_" + model_name + "_"  + strategy_name + ".xlsx"
            mape_step_name = "MAPE_Step_" + series_save_name + "_" + model_name + "_"  + strategy_name + ".xlsx"
            rmse_step_name = "RMSE_Step_" + series_save_name + "_" + model_name + "_"  + strategy_name + ".xlsx"
            running_time_name = "Running_Time_" + series_save_name + "_" + model_name + "_"  + strategy_name + ".xlsx"
            
            writer_tm = pd.ExcelWriter(file_path + "aggregate_metric/" + total_metric_name)   
            writer_ms = pd.ExcelWriter(file_path + "aggregate_metric/" + mape_step_name)
            writer_rs = pd.ExcelWriter(file_path + "aggregate_metric/" + rmse_step_name)
            writer_time = pd.ExcelWriter(file_path + "aggregate_metric/" + running_time_name)
            
            
            for H in range(2,11):       #forecast horizon 2-10：range(2,11)       #change:H = 2-10
                
                train_data_file = "../" + data_sourse_label + "/Xy/" + series_label + "/train_" + series_label + "_X" + str(F) + "_y" + str(H) + ".xlsx"
                test_data_file = "../" + data_sourse_label + "/Xy/" + series_label + "/test_" + series_label + "_X" + str(F) + "_y" + str(H) + ".xlsx"
                sea_train_data_file = "../" + data_sourse_label + "/Xy/" + type + "_s/train_" + type + "_s_X" + str(F) + "_y" + str(H) + ".xlsx"
                sea_test_data_file = "../" + data_sourse_label + "/Xy/" + type + "_s/test_" + type + "_s_X" + str(F) + "_y" + str(H) + ".xlsx" 
                
                results_list = []
                
                for t in range(TIMES):                     
                    print(series_save_name + "_" + model_name + "_" + strategy_name + "_H" + str(H) + "_time" + str(t) + "_START")  
                    
                    time_start = time.time()
                                        
                    if strategy_name == 'Dir':
                        model = []
                        selected_feature_index = []
                        best_params = []
                        for step in range(H):
                            train_score, single_model, single_feature_index, single_best_params = cp.run_clpso(model_name, strategy_name, train_data_file, F, H, train, ML_model, step)
                            
                            best_params.append(single_best_params)
                            selected_feature_index.append(single_feature_index)
                            model.append(single_model)
                    else:
                        train_score, model, selected_feature_index, best_params = cp.run_clpso(model_name, strategy_name, train_data_file, F, H, train, ML_model, step = None)
                    
                    
                    #get RMSE on training set as weight of each model and each step for BMA
                    y_train, y_train_pred, y_train_series, y_train_pred_series = test.test_model(train_data_file, model, F, H, series_label, sea_train_data_file, selected_feature_index)
                    mape_train, rmse_train, mape_train_step, mape_train_step_std, rmse_train_step, rmse_train_step_std = test.evaluate(y_train,y_train_pred, H)
#                    test.graph(y_train_series, y_train_pred_series, H, F) 
#                    print("train_total_MAPE:", mape_train)
#                    print("train_total_RMSE:", rmse_train)
                    
                    #result of test set
                    y_test, y_test_pred, y_test_series, y_test_pred_series = test.test_model(test_data_file, model, F, H, series_label, sea_test_data_file, selected_feature_index)
#                    test.graph(y_test_series, y_test_pred_series, H, F)       
                    mape, rmse, mape_step, mape_step_std, rmse_step, rmse_step_std = test.evaluate(y_test, y_test_pred, H)
                    
                    print('----------test part----------')
                    print("total_MAPE", mape)
                    print("total_RMSE", rmse)
                    print("best params:", best_params)
                    print("best selected feature index:", selected_feature_index)
                    
                    time_end = time.time()
                    print('run time(min):',(time_end - time_start)/60)
                    
                    result = {}
                    result['metric_total_train'] = np.hstack((mape_train, rmse_train)).reshape(1,-1)
                    result['metric_step_train'] = np.vstack((mape_train_step, rmse_train_step))
                    
                    result['y_test'] = y_test
                    result['y_test_pred'] = y_test_pred
                    result['y_series'] = np.concatenate((y_test_series.reshape(-1,1), y_test_pred_series.reshape(-1,1)),axis = 1) 
                    
                    result['mape'] = mape
                    result['rmse'] = rmse
                    result['metric_step'] = np.vstack((mape_step, mape_step_std, rmse_step, rmse_step_std))
                    
                    if model_name == 'MLP':
                        result['best_params'] = np.array(best_params).reshape(1,-1)
                    else:
                        result['best_params'] = np.array(best_params)
                        
                    result['selected_feature_index'] = np.array(selected_feature_index)
                    result['running_time'] = (time_end - time_start)/60
                    results_list.append(result)

                    print(series_save_name + "_" + model_name + "_" + strategy_name + "_H" + str(H) + "_time" + str(t) + "_END")
                
                
                #根据MAPE和RMSE和对重复实验的结果排序
                #sorted():python自带的排序函数，可结合operator模块，根据key进行复杂排序（此处：对所有result先按mape排序，若mape相同，再按rmse排序）
                sorted_results_list = sorted(results_list, key = itemgetter("mape", "rmse"))        
                
                #save detailed results
                column_step = ['step'+str(i+1) for i in range(H)]
                index_rank = ['rank' + str(i+1) for i in range(TIMES)]
                
                for rank, s_r in enumerate(sorted_results_list):
                    if rank < 10:                                                #change; final experiment: rank < 10 (0-9)
                        result_path = file_path + "TOP10/"                
                    else:
                        result_path = file_path
                    
                    df1 = pd.DataFrame(s_r['metric_total_train'], columns = ['mape_train','rmse_train'])  
                    df2 = pd.DataFrame(s_r['metric_step_train'], index = ['MPAE_train_step', 'RMSE_train_step'], columns = column_step)
                    df3 = pd.DataFrame(s_r['y_test'], columns = column_step)
                    df4 = pd.DataFrame(s_r['y_test_pred'], columns = column_step)
                    df5 = pd.DataFrame(s_r['y_series'], columns = ["y_test_series","y_test_series_pred"])
                    df6 = pd.DataFrame({'MAPE':[s_r['mape']],'RMSE':[s_r['rmse']]})
                    df7 = pd.DataFrame(s_r['metric_step'], index = ['MAPE_step','stdAPE_stpe','RMSE_step','stdSE_step'], columns = column_step)
                    df8 = pd.DataFrame(s_r['best_params'])
                    df9 = pd.DataFrame(s_r['selected_feature_index'])
                    
                    result_name = series_save_name + "_" + model_name + "_"  + strategy_name + "_" + "y" + str(H) + "_rank" + str(rank + 1) + ".xlsx"
                
                    writer1 = pd.ExcelWriter(result_path + result_name)
                    df1.to_excel(writer1,'metric_total_train',index = False) 
                    df2.to_excel(writer1,'metric_step_train')
                    df3.to_excel(writer1,'y_test',index = False)
                    df4.to_excel(writer1,'y_test_pred',index = False)
                    df5.to_excel(writer1,'y_series', index = False)
                    df6.to_excel(writer1,'metric_total', index = False)
                    df7.to_excel(writer1,'metric_step')
                    df8.to_excel(writer1,'best_params', index = False)
                    df9.to_excel(writer1,'selected_feature_index', index = False)
                    writer1.save() 
                    
                    print(rank)
                    print(sorted_results_list[rank]["mape"])
                    print(sorted_results_list[rank]["rmse"])
                
                #summarization of results
                MAPE_set = np.array([s_r["mape"] for s_r in sorted_results_list]).reshape(-1,1)
                RMSE_set = np.array([s_r["rmse"] for s_r in sorted_results_list]).reshape(-1,1)
                metric_set = np.concatenate((MAPE_set, RMSE_set), axis = 1)                
                df_tm = pd.DataFrame(metric_set, index = index_rank, columns = ["MAPE","RMSE"])
                df_tm.to_excel(writer_tm,'H'+str(H))
                
                mape_step_set = np.array([s_r['metric_step'][0] for s_r in sorted_results_list])
                df_ms = pd.DataFrame(mape_step_set, index = index_rank, columns = column_step)
                df_ms.to_excel(writer_ms, 'H'+str(H))
                
                rmse_step_set = np.array([s_r['metric_step'][2] for s_r in sorted_results_list])
                df_rs = pd.DataFrame(rmse_step_set, index = index_rank, columns = column_step)
                df_rs.to_excel(writer_rs, 'H'+str(H))
                
                running_time_set = np.array([s_r['running_time'] for s_r in sorted_results_list]).reshape(-1,1)
                df_time = pd.DataFrame(running_time_set, index = index_rank, columns = ['running_time (min)'])
                df_time.to_excel(writer_time, 'H'+str(H))
            
            
            writer_tm.save()
            writer_ms.save()
            writer_rs.save()
            writer_time.save()

                
total_end_time = time.time()        
print('totally run time(min):',(total_end_time - total_start_time)/60) 
    
    
    