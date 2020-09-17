#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------
# Time-series forecasting using ARIMA model


#Example:
# python arima.py

# Author: Hadia Hameed
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# ---------------------------------------------------------------

#datastructure packages
import numpy as np
import pandas as pd
from numpy import array

#file system packages
import sys
import os
from os import path
import glob
import warnings
warnings.filterwarnings('ignore')
import gzip

#datetime packages
import datetime
import time
import dateutil.parser
import pytz

#machine learning packages

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pmdarima as pm


#miscellaneous
import math
import random
import pickle

if sys.argv[-1] == '0':
    #seg = 'filtered_false/'
    seg = 'unfiltered_imputed/' #No filtering but imputing missing CGM values
    filter_data = False # median filtering
elif sys.argv[-1] == '1':
    #seg = 'filtered_true/'
    seg = 'filtered_imputed/'
    filter_data = True 
elif sys.argv[-1] == '2':
    #seg = 'filtered_false_extrapolate/'
    seg = 'unfiltered_unimputed/' #No filtering and not imputing missing CGM values
    filter_data = False 
elif sys.argv[-1] == '3':
    #seg = 'filtered_true_extrapolate/'
    seg = 'filtered_unimputed/'
    filter_data = True 

# unpickle a pickled dictionary
def unpickle_data(data_path):
    with open(data_path, 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    return unpickled_data

# re-constructs data based on single-step/multi-output, or uni/multi-variate settings
# returns X, y as "features" (historical data) and "labels" (future data)
def process_data(df):    
    included_keys = list()
    for key in df.keys():
        if 'CGM' not in key:
            included_keys.append(key)
    df.drop(included_keys, axis=1, inplace=True)
    data = df.values
    data = data.astype('float32')

    X, y = data[:,1:12], data[:, -1] #recent most glucose value to predict 30 minutes into the future.

    return X , y
#******************************* MAIN Function for CGM prediction ***************************************

def ARIMA_forecasting():
    if normalize_data:
        substring = 'normalized_'+PH+'min'
    else:
        substring = PH+'min'

    if dataset == 'oaps':
        print('Getting data from ', data_directory + seg + '\n')
        unpickled_train_data = unpickle_data(data_directory + seg + 'windowed_train_' + substring + '.pickle') #e.g. windowed_train_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + seg + 'windowed_test_' + substring + '.pickle') 
    elif dataset == 'ohio':
        unpickled_train_data = unpickle_data(data_directory + 'OhioT1DM-training/imputed/'+'windowed_' + substring + '.pickle') #e.g. windowed_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + 'OhioT1DM-testing/imputed/'+'windowed_' + substring + '.pickle')
    
    subjs = list(unpickled_train_data.keys())
    random.shuffle(subjs)

    testScores = list()
    subjects = list()
    i = 0
    
    for subj in subjs:
        i = i + 1
        print('----------Training on subject: ',subj,'----------')
        print('----------Subject: ',i,'/',len(subjs),'----------')
        df_train = unpickled_train_data[subj]
        df_test = unpickled_test_data[subj]
        df = pd.concat([df_train, df_test], axis=0)
        
        X,y = process_data(df)
        forecasts = list()
        n = int(0.2*len(X))
      
        for j in range(100,n):
            data = np.hstack(X[j-100:j])
            #model = sm.tsa.statespace.SARIMAX(X[j], trend='c', order=(1,1,0), enforce_stationarity=False, initialization='approximate_diffuse',enforce_invertibility=True)
            model = ARIMA(data, order=(1,1,0))
            #model = pm.auto_arima(data, start_p=1, start_q=1,
                    #test='adf',       # use adftest to find optimal 'd'
                    #max_p=3, max_q=3, # maximum p and q
                    #d=0,
                    #max_d=0, 
                    #m=1,              # frequency of series 
                    #seasonal=False,   # No Seasonality
                    #start_P=0,
                    #trace=True,
                    #error_action='ignore',  
                    #suppress_warnings=True)

            try:
            	model_fit = model.fit(disp=0)
            	output = model_fit.forecast(steps=6)[0]
            	yhat = output[-1]
            except:
                yhat = X[-1]
            print('----------Row: ',j,'/',n,'------Subject: ',i,'/',len(subjs),'----------')
           
            forecasts.append(yhat)
        try:
            forecasts = [ int(x) for x in forecasts ]
            error = math.sqrt(mean_squared_error(y[100:n], forecasts)) 
            print('Test RMSE: %.3f' % error)
            testScores.append(error)
            subjects.append(subj)
        except:
            continue
        

    results_df = pd.DataFrame(list(zip(subjects,testScores)),columns=['Subject','RMSE'])
    results_df.sort_values(by=['Subject'], inplace = True)      
    return results_df

def make_directories():
    #datetime_now = datetime.datetime.now()
    #datetime_now = datetime_now.strftime("%d-%m-%Y_%I-%M-%S_%p")
    if not path.exists(output_directory + 'overall_results'):
        os.mkdir(output_directory + 'overall_results')
    if not path.exists(output_directory + 'overall_results' + '/' + model_name):
        os.mkdir(output_directory + 'overall_results' + '/' + model_name)

    return output_directory + 'overall_results' + '/' + model_name
    
def main():
    results_directory = make_directories()
    overall_results = ARIMA_forecasting()
   
    if not path.exists(results_directory  + '/single_univariate_30'):
        os.mkdir(results_directory  + '/single_univariate_30')
    if not path.exists(results_directory  + '/single_univariate_30/' + seg):
        os.mkdir(results_directory  + '/single_univariate_30/' + seg)
    filename = results_directory  + '/single_univariate_30/' + seg
    if save_results:
        overall_results.to_csv(filename+'single_univariate_30min.csv')

if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        
        PH = '30' #ARIMA only predicts 30 minutes into the future
        if sys.argv[4] == 'False':
            normalize_data = False
        else:
            normalize_data = True
        model_name = sys.argv[5]
        dataset = sys.argv[6]
        if sys.argv[7] == 'False':
            save_results = False
        else:
            save_results = True
        
        print("\n Starting experiments for the following settings:\n normalize_data: "+str(normalize_data)+";\n model_name: "+model_name+" ;\n dataset: "+dataset+";\n save_results: "+str(save_results)+';\n ablation code: '+seg+'\n')
        main()
        print("\n Experiments completed for the following settings:\n normalize_data: "+str(normalize_data)+";\n model_name: "+model_name+" ;\n dataset: "+dataset+";\n save_results: "+str(save_results)+';\n ablation code: '+seg+'\n')
    else:
        print("Invalid input arguments")
        exit(-1)
