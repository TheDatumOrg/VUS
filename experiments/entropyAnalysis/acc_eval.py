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


import warnings



def generate_data(filepath,init_pos):
    
    df = pd.read_csv(filepath, header=None).to_numpy()
    name = filepath.split('/')[-1]
    data = df[0:len(df),0].astype(float)
    label = df[0:len(df),1]
    
    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()

    data_train = data[:int(0.1*len(data))]
    data_test = data

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    
    return slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label

def compute_score(slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label):
        
    methods = [
            'NormA',
            'POLY',
            'IForest',
            'AE',
            'OCSVM',
            'MatrixProfile',
            'LOF',
            'LSTM',
            #'CNN',
    ]
    
    methods_acc = {}
    for methods_score in methods:
        methods_acc[methods_score] = {}
        methods_acc[methods_score]['R_AUC_ROC']      =[]
        methods_acc[methods_score]['AUC_ROC']        =[]
        methods_acc[methods_score]['R_AUC_PR']       =[]
        methods_acc[methods_score]['AUC_PR']         =[]
        methods_acc[methods_score]['VUS_ROC']        =[]
        methods_acc[methods_score]['VUS_PR']         =[]
        methods_acc[methods_score]['Precision']      =[]
        methods_acc[methods_score]['Recall']         =[]
        methods_acc[methods_score]['F']              =[]
        methods_acc[methods_score]['ExistenceReward']=[]
        methods_acc[methods_score]['OverlapReward']  =[]
        methods_acc[methods_score]['Precision@k']    =[]
        methods_acc[methods_score]['Rprecision']     =[]
        methods_acc[methods_score]['Rrecall']        =[]
        methods_acc[methods_score]['RF']             =[]
    
    for iter_rand in tqdm(range(5)):
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
            print("computing {}".format(method))
            methods_scores[method] = score
        
        print(methods)
        for methods_score in tqdm(methods):
            print("eval {}".format(methods_score))
            new_label = label
            
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=methods_scores[methods_score], window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, methods_scores[methods_score], plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, methods_scores[methods_score])  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,methods_scores[methods_score],2*slidingWindow)
            L1 = [ elem for elem in L]

            methods_acc[methods_score]['R_AUC_ROC']      +=[R_AUC]
            methods_acc[methods_score]['AUC_ROC']        +=[L1[0]]
            methods_acc[methods_score]['R_AUC_PR']       +=[R_AP]
            methods_acc[methods_score]['AUC_PR']         +=[AP]
            methods_acc[methods_score]['VUS_ROC']        +=[avg_auc_3d]
            methods_acc[methods_score]['VUS_PR']         +=[avg_ap_3d]
            methods_acc[methods_score]['Precision']      +=[L1[1]]
            methods_acc[methods_score]['Recall']         +=[L1[2]]
            methods_acc[methods_score]['F']              +=[L1[3]]
            methods_acc[methods_score]['ExistenceReward']+=[L1[5]]
            methods_acc[methods_score]['OverlapReward']  +=[L1[6]]
            methods_acc[methods_score]['Precision@k']    +=[L1[9]]
            methods_acc[methods_score]['Rprecision']     +=[L1[7]]
            methods_acc[methods_score]['Rrecall']        +=[L1[4]]
            methods_acc[methods_score]['RF']             +=[L1[8]]                
    return methods_acc


def generate_curve(label,score,slidingWindow):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d



def run_method_analysis(filepath):

    
    
    slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label = generate_data(filepath,0)
    if slidingWindow is None:
        print("[ABORT]",filepath)
        return None
    print(filepath,slidingWindow)
    methods_acc = compute_score(slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label)
    
    with open('../../results/acc_eval/{}_acc.txt'.format(filepath.split('/')[-1]), 'w') as fp:
        print(methods_acc, file=fp)
    
    return methods_acc


def multi_run_wrapper(args):
   return run_method_analysis(*args)

def main():

    all_files = []
    
    for folder in os.listdir('../../benchmark/'):
        if '.' not in folder:
            for file_s in os.listdir('../../benchmark/{}/'.format(folder)):
                all_files.append(['../../benchmark/{}/{}'.format(folder,file_s)])

    res = run_method_analysis(all_files[0][0])    
    #with Pool(processes=1) as pool:
    #    results = pool.map(multi_run_wrapper,all_files)    
    



if __name__ == '__main__':
        main()



