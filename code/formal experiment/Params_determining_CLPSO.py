#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[]:
def CLPSO_params(model_name, strategy_name, F):
    maxIter = 200     #最大迭代次数
    maxFailIter = 30  #最大连续且无improve的迭代次数
    C = 2             #C:学习因子
    w_max = 0.9       #惯性权重w 初始值（max）
    w_min = 0.4       #惯性权重w 终止值（min）
    N = 8             #粒子个数
    D = F + modelParam_to_particle(model_name, strategy_name)[0]    #每个粒子的维度：F(features) = 20 + 超参0-1编码所占的维数
    M = 8             #refreshing gap
    
    return maxIter, maxFailIter, C, w_max, w_min, N, D, M


# In[]:
#设置各模型中待调参数的参数范围及名称；参数名称与sklearn库中各模型参数名称一致
def model_params_init(model_name, strategy_name):      
    if model_name =='SVR':
#        param_range = {'C':np.logspace(-2, 2, 32), 'epsilon':np.logspace(-4, 0, 4), 'gamma':[0.05, 0.1, 0.2, 0.4]}   #使MSVR在S_ILI上表现较好：需要较大的C值，较小的gamma值;   
        param_range = {'C':np.logspace(0, 2, 16), 'epsilon':np.logspace(-4, -2, 4), 'gamma':[0.05, 0.1, 0.2, 0.4]}
        param_keys = ['C','epsilon', 'gamma']     
    elif model_name == 'MLP':
#        param_range = {'hidden_layer_sizes':[(2,), (4,), (6,), (8,), (10,), (20,), (50,), (100,)]}
        param_range = {'hidden_layer_sizes':[(50,), (100,)]}
        param_keys = ['hidden_layer_sizes']
    elif model_name == 'KNN':
        param_range = {'n_neighbors':np.linspace(2,32,16).astype(int)}      # neighbor数K = [1,2,3,...,32],32个值
        param_keys = ['n_neighbors']
        
    return param_range, param_keys

#根据参数的调参范围，计算各参数对应在粒子中所占的维数
def modelParam_to_particle(model_name, strategy_name):   
    D_param_part = 0
    param_range, param_keys = model_params_init(model_name, strategy_name)
    bin_length = {}    
    for p_n in param_keys:
        bin_length[p_n] = np.log2(len(param_range[p_n])).astype(int)        #粒子中各参数的维数 = log2(调参范围给出的参数个数)
        D_param_part += bin_length[p_n]
        
    return D_param_part, bin_length

#将粒子中的0-1表示还原回对应的待调参数值，并加入其它非调节参数。返回sklearn模型的所有参数组合和待调参数组合
def particle_to_modelParam(p_i, model_name, strategy_name):
    param_range, param_keys = model_params_init(model_name, strategy_name)
    bin_length = modelParam_to_particle(model_name, strategy_name)[1]
    param_bin = {}      #粒子中各参数的0-1表示部分
    param_restore = {}     #还原后的参数  

    index_start = 0
    for p_n in param_keys:
        param_bin[p_n] = p_i[index_start:(index_start + bin_length[p_n])].copy()
        index_start = index_start + bin_length[p_n]                                                                   #下一个参数的bin维度起点：当前参数的终点
        param_restore[p_n] = param_range[p_n][param_bin[p_n].dot(2**np.arange(param_bin[p_n].size)[::-1])]            #模型中的参数由粒子的0-1表示复原回对应的实际值
    
    params_dict = param_restore.copy()   # 加入模型中需要的其他参数（不由CLPSO进行调节的参数）
    if model_name == 'SVR':
        params_dict['kernel'] = 'rbf'
#        params_dict['gamma'] = 'auto'   #defalut in sklearn, gamma=1/n_features.（因有MSVR，此项仍需手动设置和调参）
    elif model_name == 'MLP':
        params_dict['solver'] = 'lbfgs'
        params_dict['max_iter'] = 500
#        params_dict['tol'] = 1e-3
    elif model_name == 'KNN':
        params_dict['algorithm'] = 'brute'
        params_dict['weights'] = 'distance'
        params_dict['n_jobs'] = -1
    
    return params_dict, list(param_restore.values())        

#切出粒子特征选择部分的维度。（特征不需要还原，向train输入时保持0-1表示即可，train中提取特征时可直接使用0-1 index，还原是为了输出直观）
def particle_to_features(p_i, F):     
    #避免特征部分的粒子维数均为0（无入选特征）：若出现此情形，则设置特征的最后一位掩码为1
    if p_i[-F:].all() == 0:
        p_i[-1] = 1         #在p_i上改变，改变被记录在粒子上。
    
    feature_mask = p_i[-F:].copy()
    
    return feature_mask, p_i
# In[]:

def run_clpso(model_name, strategy_name, train_data_file, F, H, train, ML_model, step = None):
    #artificial parameters for PSO
    maxIter, maxFailIter, C, w_max, w_min, N, D, M = CLPSO_params(model_name, strategy_name, F)
    
    #initialization p, v, pbest
    p = np.random.randint(0,2,(N, D))    #随机生成只包含0或1的N*D的矩阵（1个粒子->D个维度; N个粒子）
    v = np.zeros((N, D))
    #pbest = p                           #very important!!!!! b = a:没有创建新的numpy对象（array）！！！！！！！！！！！！
    pbest = p.copy()
    
    t = 0
    t_useless = 0
    
    fit_p = np.zeros((N,1))             #fit_p:每个粒子每轮计算fitness function得到的值
    fit_pbest = np.zeros((N,1))         #fit_pbest: N * 1
    fit_gbest = 1                       #fit_gbest: 值
    
    refresh_gap = np.full((N,1),M+1)    #初始：需要给每个粒子分配exemplary 即每个粒子M > 8 
    exemplar = np.zeros((N,D))          #每个粒子每个维度的exemplary的index                    
    
    s_v = np.zeros((N,D))               #sigmoid function of v
    
    pc = np.zeros((N,1))                #predefined probability(Pc(i))
    for i in range(N):
        pc[i] = 0.05 + 0.45*(np.exp(10*(i-1)/(N-1))-1)/(np.exp(10)-1)
    
    fit_gbest_all = []
    
    best_model = None 
    selected_feature_index = None 
    best_params = None
    
    #evaluate the swarm and initialize the gbest
    for i in range(N):
        #train model 
        feature_mask, p[i] = particle_to_features(p[i], F)
        params_dict = particle_to_modelParam(p[i], model_name, strategy_name)[0]
        fit_p[i] = train.training_model(train_data_file, ML_model, feature_mask, params_dict, F, H, step)[0]             #函数返回多个值：实际返回的是一个值组成的tuple，可以根据tuple的index挑选需要的返回值
        #fit_p[i] = score
        fit_pbest[i] = fit_p[i].copy()
        
    fit_gbest = fit_pbest.min().copy()
    gbest = pbest[fit_pbest.argmin()].copy()             
    
    #print("gbest_init:", gbest)
    #print("--------------------------------------------------------------------")
    
    while(t < maxIter and t_useless < maxFailIter):
        
        w = ((w_max - w_min)*(maxIter-t))/maxIter + w_min
        p_not_updated = list(range(N))                           #未更新velocity的粒子（index）
        
        for i in range(N):            
            #assign exemplar for each d
            if refresh_gap[i] > M:
#                own_exemplar = 0
                if(len(p_not_updated) > 1):                         #list():()中的元素不能为空（[]）,否则报错：'NoneType' object is not iterable
                    p_candidate = list(p_not_updated)               #不能直接list().remove(i),否则p_candidate为None
                    p_candidate.remove(i)
                    #p_candidate2 = list(p_candidate)               #python中复制list不能直接用“=”，否则指向同一对象
                else:
                    p_candidate = []
                    
                for d in range(D):
                    if random.random() > pc[i]:
                        exemplar[i][d] = i
#                        own_exemplar += 1
                    else:
                        if(p_candidate != []):
                            #print("p_not_updated:",p_not_updated)
                            #print("p_candidate",p_candidate)
                            p1_i = random.choice(p_candidate)          #random.choice:从一个list中随机挑选一个元素                            
                            
                            p_candidate2 = list(p_candidate).copy()    #python中复制list不能直接用“=”，否则指向同一对象
                            if len(p_candidate) > 1:
                                p_candidate2.remove(p1_i)                           
                            p2_i = random.choice(p_candidate2)
                            
                            exemplar[i][d] = p1_i if fit_pbest[p1_i] < fit_pbest[p2_i] else p2_i
                        else:
                            exemplar[i][d] = i
#                            own_exemplar += 1
                if exemplar[i].all() == i:                                             #np.all():ndarray中的所有元素，可以与单个值作比较。
                    exemplar[i][random.randint(0,D-1)] = random.randint(0,N-1)         #numpy中的random.randint():取首不取尾；random模块中的random.randint():首尾均取
                    
            
            #print("particle index:",i)
            
            exemplar_int = exemplar.astype(int)             #得到的exemplar数组中数据类型为float，须转换为int才能作为index
            #print("exemplar:",exemplar_int)
            
            
            #print(str(i)+":gbest"+"_before update v:", gbest)
            
            #update v,p
            for d in range(D):                
                v[i][d] = w*v[i][d] + C * random.random() * (pbest[exemplar_int[i][d]][d] - p[i][d])  #random.random():0-1之间的随机数
                s_v[i][d] = 1 / (1 + np.exp(-v[i][d]))
                p[i][d] = 1 if random.random() < s_v[i][d] else 0
            
            #！！！update v和p前后gbest发生变化：表示只要p[i]变，牵一发而动全身,导致pbest和gbest异常
            
            #p[i]:updated
            p_not_updated.remove(i)
            
            #print(str(i)+":gbest"+"_before update p and pbest:", gbest)
            
            #calculate fitness function and update pbest,fit_pbest
            #train model 
            feature_mask, p[i] = particle_to_features(p[i], F)
            params_dict = particle_to_modelParam(p[i], model_name, strategy_name)[0]
            fit_p[i] = train.training_model(train_data_file, ML_model, feature_mask, params_dict, F, H, step)[0]  #函数返回多个值：实际返回的是一个值组成的tuple，可以根据tuple的index挑选需要的返回值
            
            if fit_p[i] < fit_pbest[i]:
                pbest[i] = p[i].copy()
                fit_pbest[i] = fit_p[i].copy()
                refresh_gap[i] = 0 
            else:
                refresh_gap[i] += 1                                                   
            
            #print(str(i)+":gbest"+"_after update p and pbest:", gbest)
            
        #update gbest,fit_gbest
        if fit_pbest.min() < fit_gbest:
            #print("updated")
            
            gbest = pbest[fit_pbest.argmin()].copy()
            
            fit_gbest = fit_pbest.min().copy()
            
            #print("fit_gbest:", fit_gbest)
            #print("iteration finished:",t)
            #print("pbest_min:",pbest[fit_pbest.argmin()])
            #print("fit_pbest_min:",fit_pbest.min())           
            
            #print("gbest:",gbest)
            #print("new_fit_gbest:", fit_gbest)
            #print("selected_feature_index:",selected_feature_index)
            #print("best_params:",best_params)
            
            #print("---------------------------------")
            
            t_useless = 0
        else:
            #print("not updated")            
            
            
#            print("pbest_min:",pbest[fit_pbest.argmin()])
#            print("fit_pbest_min:",fit_pbest.min())
#            
#            print("gbest:",gbest)
            
            #print("---------------------------------")
            
            t_useless +=1
        
        fit_gbest_all.append(fit_gbest)
        
        #print("gbest(iterfin):",gbest)
        #print("fit_gbest:", fit_gbest)
        #print("fit_pbest:\n", fit_pbest)
        #print("examplar:\n",exemplar.astype(int))
        #print("iteration finished:",t)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        
        t += 1
    
    feature_mask, gbest = particle_to_features(gbest, F)
    params_dict, best_params = particle_to_modelParam(gbest, model_name, strategy_name)
    fit_gbest, best_model, selected_feature_index = train.training_model(train_data_file, ML_model, feature_mask, params_dict, F, H, step)
    
#    print("**********************************************************************************")
#    print("gbest_final:",gbest)
#    print("fit_gbest:",fit_gbest)
#    print("selected_feature_index:", selected_feature_index)
#    print("best_params:",best_params)
#    print("total iteration times:",t)
    
#    plt.plot(fit_gbest_all)
#    plt.show()    
    
    return fit_gbest, best_model, selected_feature_index, best_params


