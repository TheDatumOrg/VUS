import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import random
from random import shuffle
from matplotlib import cm
from multiprocessing import Pool
import pickle

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from vus.models.distance import Fourier
from vus.models.feature import Window
from vus.utils.slidingWindows import find_length
from vus.utils.metrics import metricor
from vus.models.cnn import cnn
from vus.models.AE_mlp2 import AE_MLP2
from vus.models.lstm import lstm
from vus.models.ocsvm import OCSVM
from vus.models.poly import POLY
from vus.models.pca import PCA
from vus.models.norma import NORMA
from vus.models.matrix_profile import MatrixProfile
from vus.models.lof import LOF
from vus.models.iforest import IForest


def find_section_length(label,length):
    best_i = None
    best_sum = None
    current_subseq = False
    for i in range(len(label)):
        changed = False
        if label[i] == 1:
            if current_subseq == False:
                current_subseq = True
                if best_i is None:
                    changed = True
                    best_i = i
                    best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                else:
                    if np.sum(label[max(0,i-200):min(len(label),i+9800)]) < best_sum:
                        changed = True
                        best_i = i
                        best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                    else:
                        changed = False
                if changed:
                    diff = i+9800 - len(label)
            
                    pos1 = max(0,i-200 - max(0,diff))
                    pos2 = min(i+9800,len(label))
        else:
            current_subseq = False
    if best_i is not None:
        return best_i-pos1,(pos1,pos2)
    else:
        return None,None

def generate_data(filepath,init_pos,max_length):
    
    df = pd.read_csv(filepath, header=None).to_numpy()
    name = filepath.split('/')[-1]
    data = df[0:len(df),0].astype(float)
    label = df[0:len(df),1]
    
    pos_first_anom,pos = find_section_length(label,max_length)
    if pos is None:
        return None,None,None,None,None,None,None,None,None
    data = df[pos[0]:pos[1],0].astype(float)
    label = df[pos[0]:pos[1],1]
    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()

    data_train = data[:int(0.1*len(data))]
    data_test = data

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    
    return pos_first_anom,slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label

def compute_score(slidingWindow,data,X_data,data_train,data_test,X_train,X_test):
        
        methods = [
                'NormA',
                'POLY',
                'IForest',
                'AE',
                'OCSVM',
                'MatrixProfile',
                'LOF',
                'LSTM',
                'CNN',
        ]

        methods_scores = {}
        for method in methods:
                start_time = time.time()
                if method == 'IForest':
                        clf = IForest(n_jobs=1)
                        x = X_data
                        clf.fit(x)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

                elif method == 'LOF':
                        clf = LOF(n_neighbors=20, n_jobs=1)
                        x = X_data
                        clf.fit(x)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

                elif method == 'MatrixProfile':
                        clf = MatrixProfile(window = slidingWindow)
                        x = data
                        clf.fit(x)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

                elif method == 'NormA':
                        clf = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow)
                        x = data
                        clf.fit(x)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                        score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

                elif method == 'PCA':
                        clf = PCA()
                        x = X_data
                        clf.fit(x)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

                elif method == 'POLY':
                        clf = POLY(power=3, window = slidingWindow)
                        x = data
                        clf.fit(x)
                        measure = Fourier()
                        measure.detector = clf
                        measure.set_param()
                        clf.decision_function(measure=measure)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

                elif method == 'OCSVM':
                        X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
                        X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
                        clf = OCSVM(nu=0.05)
                        clf.fit(X_train_, X_test_)
                        score = clf.decision_scores_
                        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

                elif method == 'LSTM':
                        clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 50, patience = 5, verbose=0)
                        clf.fit(data_train, data_test)
                        measure = Fourier()
                        measure.detector = clf
                        measure.set_param()
                        clf.decision_function(measure=measure)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

                elif method == 'AE':
                        clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0)
                        clf.fit(data_train, data_test)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

                elif method == 'CNN':
                        clf = cnn(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 100, patience = 5, verbose=0)
                        clf.fit(data_train, data_test)
                        measure = Fourier()
                        measure.detector = clf
                        measure.set_param()
                        clf.decision_function(measure=measure)
                        score = clf.decision_scores_
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

                end_time = time.time()
                time_exec = end_time - start_time
                methods_scores[method] = score
                
        return methods_scores



def generate_new_label(label,lag):
    if lag < 0:
        return np.array(list(label[-lag:]) + [0]*(-lag))
    elif lag > 0:
        return np.array([0]*lag + list(label[:-lag]))
    elif lag == 0:
        return label

def bounded_random_walk(length, lower_bound,  upper_bound, start, end, std):
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random(length) - 0.5)).cumsum()
    rand_trend = np.linspace(rand[0], rand[-1], length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.max([1, (rand_deltas.max()-rand_deltas.min())/bounds])

    trend_line = np.linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_line
    lower_bound_delta = lower_bound - trend_line

    upper_slips_mask = (rand_deltas-upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_line + rand_deltas



    
def generate_curve(label,score,slidingWindow):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d


def compute_anomaly_acc_lag(methods_scores,label,slidingWindow,methods_keys):
    
    lag_range = list(range(-slidingWindow//4,slidingWindow//4,max(1,(slidingWindow//2)//20) ))
    methods_acc = {}
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for lag in lag_range:
            new_label = generate_new_label(label,lag)
            
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=methods_scores[methods_score], window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, methods_scores[methods_score], plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, methods_scores[methods_score])  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,methods_scores[methods_score],2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc


def compute_anomaly_acc_percentage(methods_scores,label,slidingWindow,methods_keys,pos_first_anom):
    
    list_pos = []
    step_a = max(0,(len(label) - pos_first_anom-200))//20
    step_b = max(0,pos_first_anom-200)//20
    pos_a = min(len(label),pos_first_anom + 200)
    pos_b = max(0,pos_first_anom - 200)
    list_pos.append((pos_b,pos_a))
    for pos_iter in range(20):
        pos_a = min(len(label),pos_a + step_a)
        pos_b = max(0,pos_b - step_b)
        list_pos.append((pos_b,pos_a))

    
    methods_acc = {}
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for end_pos in list_pos:
            new_label = label[end_pos[0]:end_pos[1]]
            new_score = np.array(methods_scores[methods_score])[end_pos[0]:end_pos[1]]
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_score, window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, new_score, plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, new_score)  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_score,2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc

def compute_anomaly_acc_noise(methods_scores,label,slidingWindow,methods_keys):
    
    lag_range = list(range(-slidingWindow//2,slidingWindow//2,10))
    methods_acc = {}
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for lag in range(20):
            new_label = label
            
            grader = metricor()  

            noise = np.random.normal(-0.1,0.1,len(methods_scores[methods_score]))
            
            new_score = np.array(methods_scores[methods_score]) + noise
            new_score = (new_score - min(new_score))/(max(new_score) - min(new_score))
            
            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_score, window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, new_score, plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, new_score)  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_score,2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc

def box_plot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp



def normalize_dict_exp(methods_acc_lag):
    key_metrics = [
        'VUS_ROC',
        'VUS_PR',
        'R_AUC_ROC',
        'R_AUC_PR',
        'AUC_ROC',
        'AUC_PR',
        'Rprecision',
        'Rrecall',
        'RF',
        'Precision',
        'Recall',
        'F',
        'Precision@k'
    ]
    methods_keys = [
        'NormA',
        'POLY',
        'IForest',
        'AE',
        'OCSVM',
        'MatrixProfile',
        'LOF',
        'LSTM',
        'CNN',
    ]
    norm_methods_acc_lag = {}
    for key in methods_keys:
        norm_methods_acc_lag[key] = {}
        for key_metric in key_metrics:
            ts = methods_acc_lag[key][key_metric]
            new_ts = list(np.array(ts) -  np.mean(ts))
            norm_methods_acc_lag[key][key_metric] = new_ts
    return norm_methods_acc_lag
        
def group_dict(methods_acc_lag):
    key_metrics = [
        'VUS_ROC',
        'VUS_PR',
        'R_AUC_ROC',
        'R_AUC_PR',
        'AUC_ROC',
        'AUC_PR',
        'Rprecision',
        'Rrecall',
        'RF',
        'Precision',
        'Recall',
        'F',
        'Precision@k'
    ]
    methods_keys = [
        'NormA',
        'POLY',
        'IForest',
        'AE',
        'OCSVM',
        'MatrixProfile',
        'LOF',
        'LSTM',
        'CNN',
    ]
    norm_methods_acc_lag = {key:[] for key in key_metrics}
    for key in methods_keys:
        for key_metric in key_metrics:
            ts = list(methods_acc_lag[key][key_metric])
            new_ts = list(np.array(ts) -  np.mean(ts))
            norm_methods_acc_lag[key_metric] += new_ts
    return norm_methods_acc_lag



def run_method_analysis(filepath):
    #try:
    methods_keys = [
        'NormA',
        'POLY',
        'IForest',
        'AE',
        'OCSVM',
        'MatrixProfile',
        'LOF',
        'LSTM',
        'CNN',
    ]

    print(filepath)
    max_length = 10000
    pos_first_anom,slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label = generate_data(filepath,0,max_length)
    if slidingWindow is None:
        return None
    methods_scores =  compute_score(slidingWindow,data,X_data,data_train,data_test,X_train,X_test)
    
    methods_acc_lag = compute_anomaly_acc_lag(methods_scores,label,slidingWindow,methods_keys)
    methods_acc_noise = compute_anomaly_acc_noise(methods_scores,label,slidingWindow,methods_keys)
    methods_acc_percentage = compute_anomaly_acc_percentage(methods_scores,label,slidingWindow,methods_keys,pos_first_anom)

    norm_methods_acc_lag = normalize_dict_exp(methods_acc_lag)
    norm_methods_acc_noise = normalize_dict_exp(methods_acc_noise)
    norm_methods_acc_percentage = normalize_dict_exp(methods_acc_percentage)

    group_norm_methods_acc_lag = group_dict(methods_acc_lag)
    group_norm_methods_acc_noise = group_dict(methods_acc_noise)
    group_norm_methods_acc_percentage = group_dict(methods_acc_percentage)


    all_res_robust = {}
    all_res_robust_lag = {}
    all_res_robust_noise = {}
    all_res_robust_percentage = {}
    for key in group_norm_methods_acc_lag.keys():
        std_1 = np.std(group_norm_methods_acc_lag[key])
        std_2 = np.std(group_norm_methods_acc_noise[key])
        std_3 = np.std(group_norm_methods_acc_percentage[key])

        all_res_robust_lag[key] = std_1
        all_res_robust_noise[key] = std_2
        all_res_robust_percentage[key] = std_3
        all_res_robust[key] = np.mean([std_1,std_2,std_3])
    #print(all_res_robust)
    #with open('result_data_raw/{}_accuracy_percentage_group.pickle'.format(filepath.split('/')[-1]), 'wb') as fp:
    #    pickle.dump([group_norm_methods_acc_lag,group_norm_methods_acc_noise,group_norm_methods_acc_percentage], fp)
    #with open('result_data_raw/{}_accuracy_percentage.pickle'.format(filepath.split('/')[-1]), 'wb') as fp:
    #    pickle.dump([norm_methods_acc_lag,norm_methods_acc_noise,norm_methods_acc_percentage], fp)

    

    with open('../../results/robustness_results/result_data_aggregated_lag/{}_robustness.txt'.format(filepath.split('/')[-1]), 'w') as fp:
        print(all_res_robust_lag, file=fp)
    with open('../../results/robustness_results/result_data_aggregated_noise/{}_robustness.txt'.format(filepath.split('/')[-1]), 'w') as fp:
        print(all_res_robust_noise, file=fp)
    with open('../../results/robustness_results/result_data_aggregated_percentage/{}_robustness.txt'.format(filepath.split('/')[-1]), 'w') as fp:
        print(all_res_robust_percentage, file=fp)

    #with open('result_data_aggregated/{}_robustness.txt'.format(filepath.split('/')[-1]), 'w') as fp:
    #    print(all_res_robust, file=fp)

    return all_res_robust


def multi_run_wrapper(args):
   return run_method_analysis(*args)

def main():

    methods = [
        'NormA',
        'POLY',
        'IForest',
        'AE',
        'OCSVM',
        'MatrixProfile',
        'LOF',
        'LSTM',
        'CNN',
    ]

    all_files = []
    
    for folder in os.listdir('../../benchmark/'):
        if '.' not in folder:
            for file_s in os.listdir('../../benchmark/{}/'.format(folder)):
                all_files.append(['../../benchmark/{}/{}'.format(folder,file_s)])

    with Pool(processes=4) as pool:
        results = pool.map(multi_run_wrapper,all_files)    
    



if __name__ == '__main__':
        main()



