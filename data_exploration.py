#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------
# All the files in /data/PHI/PHI_OAPS/sandbox/hhameed/data_info
# were created using the following functions
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

#plotting packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#datetime packages
import datetime
import time
import dateutil.parser

#local modules
import preprocess as PR
import helpers as HL

#miscellaneous
import math
from operator import add
import json

#get path to the main directory and data files
oaps_data = 'n=88_OpenAPSDataAugust242018'
#oaps_data = '(n=1)*111OpenAPSDataCommonsJan2019'

def get_files(oaps_data_folder):
    cwd = os.getcwd()
    nested_directories = cwd.count('/') #no. of directories to go back
    root = ''
    for i in range(nested_directories):
        root = root+"../"

    root_directory = root+"data/PHI/PHI_OAPS/"
    data_directory = "%sOpenAPS_data/%s/" %(root_directory,oaps_data_folder)
    subjs = [subject for subject in os.listdir(data_directory) if "." not in subject]
    subjs.sort()
    return root_directory,data_directory,subjs


root_directory,data_directory,subjs = get_files()

#grouping data according to year, month, day. "i" is the nth year/month/day in the group
#e.g: if a subject has data for Jan-Dec 2017 and Jan-Dec 2018, and i = 0 with flag ='all', 
# it will return data for all the days of Jan 2017 (day_groups), along with data for all 
# the years (year_groups) and all the months for 2017 (month_groups)
# if flag = 'year', it will only group by years
def group_data(df,flag='year',i=0):
    if flag == 'all':
        #group by year
        year_groups = df.groupby(df['year'])
        year = year_groups.keys.unique()[i] #get the ith year's data
        year_group = year_groups.get_group(year)

        #group by month
        year_group['month'] = [d.month for d in year_group['date']] #get all the months in a separate column
        month_groups = year_group.groupby(year_group['month'])

        #group by day
        month = month_groups.keys.unique()[i] #get the ith month's data
        month_group = month_groups.get_group(month)
        month_group['day'] = [d.day for d in month_group['date']]  #get all the days in a separate column
        day_groups = month_group.groupby(month_group['day'])

        return year_groups,month_groups,day_groups
    
    elif flag == 'year':
        #group by year
        year_groups = df.groupby(df['year'])
        return year_groups

        

def get_subject_data(subj, feature,oaps_data):
    _,data_directory,_ = get_files(oaps_data)
    data_frames = list()
    file_names = list()
    subj_data_file = '%s%s/direct-sharing-31/extracted_files/' %(data_directory,subj)
    if path.exists(subj_data_file) == True:
        files = glob.glob(subj_data_file+feature+"*.csv")
        files.sort()
        for f in files:
            df = pd.read_csv(f)
            if not df.empty:
                file_names.append(f.split('/')[-1]) #e.g. entries.csv
                if feature == 'entries':
                    df.drop_duplicates(subset=['dateString','sgv'], keep='first', inplace=True)
                    data_frames.append(df[['dateString','sgv','device']])
                if feature == 'devicestatus' and 'openaps' in df.columns:
                    data_frames.append(df[['openaps']])
    return data_frames,file_names


#compare files in the two OAPS datasets
def compare_data_sets():
    features = ['entries', 'devicestatus', 'profile','treatments']
    columns=['Subject','No. of new entries files','No. of new device files','No. of new profile files', 'No. of new treatment files']
    data_info = pd.DataFrame(columns=columns)
    root_directory1, data_directory1, subjs1 = get_files('n=88_OpenAPSDataAugust242018')
    root_directory2, data_directory2, subjs2 = get_files('(n=1)*111OpenAPSDataCommonsJan2019')
    
    common_subjs = [subj for subj in subjs1 if subj in subjs2]
    
    common_subjs.sort()
    for subj in common_subjs:
        print(subj)
        
        n = list()
        for feature in features:
            print(feature)
            data_frames1,file_names1 = get_subject_data(subj,feature,'n=88_OpenAPSDataAugust242018')
            data_frames2,file_names2 = get_subject_data(subj,feature,'(n=1)*111OpenAPSDataCommonsJan2019')
            difference = list(set(file_names2) - set(file_names1))
            for g in file_names2:
                if (g in file_names1):
                    difference = list(set(file_names2) - set(file_names1))
            print(difference)
            n.append(len(difference))
        print(n)
        d = pd.DataFrame([[subj,n[0],n[1],n[2],n[3]]],columns=columns)
        data_info = pd.concat([d,data_info])

    data_info = data_info.sort_values(by=[columns[0]]) #sort by subject
    filename = '%s/sandbox/hhameed/data_info/' %root_directory #save in /data/PHI/PHI_OAPS/sandbox/hhameed/data_info/ 
    #uncomment to replace files 
    #data_info.to_csv(filename+'dataset_comparison.csv') 
    
#how many subjects have predicted BGs recorded in their data
def get_subjects_with_predicted_BGS():
    feature = 'devicestatus'
    columns=['Subject','File','predicted_values_absent (%)']
    data_info = pd.DataFrame(columns=columns)

    for subj in subjs:
        subj_data_file = '%s%s/direct-sharing-31/extracted_files/' %(data_directory,subj)
        if path.exists(subj_data_file) == True:
            files = glob.glob(subj_data_file+feature+"*.csv")
            for f in files:
                df = pd.read_csv(f)
                if 'openaps' in df.columns:
                    print(subj)
                    missing_values = float(sum(pd.isnull(df['openaps'])))/float(df.shape[0])
                    missing_values_percent = round(100*missing_values, 2)               
                    d = pd.DataFrame([[subj,f.split('/')[-1],missing_values_percent]],columns=columns)
                    data_info = pd.concat([d,data_info])
    
    filename = '%s/sandbox/hhameed/data_info/' %root_directory #save in /data/PHI/PHI_OAPS/sandbox/hhameed/data_info/    
    #uncomment to replace files
    #data_info.to_csv(filename+'predicted_BGS_info.csv') 

#get the following information about each year of data for each subject
# - subject name, filename, year, total number of samples, total length of recording in days, percentage of missing CGM values
def data_statistics():
    columns=['Subject','File','Year','No. of samples','Length of recording (days)','%% of missing CGM values']
    data_info = pd.DataFrame(columns=columns)
    missing_data_info = pd.DataFrame(columns=[columns[0],columns[1],'missing_value_percentage'])
    columns2=['Subject','Year','No. of months present','Missing months','No. of samples','Length of recording (days)']
    complete_data_info = pd.DataFrame(columns=columns2)

    for subj in subjs:
        data_frames,file_names = get_subject_data(subj, 'entries',oaps_data)
        combined_data = pd.DataFrame()
        print("Getting data info for subject: " + subj + "\n")
        for df, file_name in zip(data_frames,file_names):
            combined_data = pd.concat([df,combined_data])
            missing_data_percentage = df.isnull().mean().round(4) * 100 #gives percentage of missing values in each column
            
            grouped = group_data(df,'year') #grouped according to years

            #rows with missing dates OR CGM values
            missing_data = pd.DataFrame([[subj,file_name,missing_data_percentage]],columns=[columns[0],columns[1],'missing_value_percentage'])
            missing_data_info = pd.concat([missing_data,missing_data_info])

            #get information about the data for each year for a particular subject
            for name, group in grouped:
                
                #rows with missing CGM values in each year for a specific subject
                missing_cgm = float(sum(pd.isnull(group['sgv'])))/float(group.shape[0])
                missing_cgm_percent = round(100*missing_cgm, 2)

                d = pd.DataFrame([[subj,file_name,name,group.shape[0],int((5.0/(24*60.0))*group.shape[0]),missing_cgm_percent]],columns=columns)
                data_info = pd.concat([d,data_info])
        
        #combine data from different files for a given subject and store it as a single file
        subj_combined_data = combined_data.sort_values(by='dateString')
        extracted_folder_dir = '%s%s/direct-sharing-31/extracted_files/complete_data_%s.csv' %(data_directory,subj,subj)
        subj_combined_data.to_csv(extracted_folder_dir)
        
        #group all the data according to year and month
        grouped = group_data(combined_data,'year')
        
        for name, group in grouped:
            group['month'] = [d.month for d in group['date']]  #group each year by month 
            grouped_by_month = group.groupby(group['month'])
            months = list()
            days = list()
            for name_month,group_month in grouped_by_month:
                months.append(name_month)
                group_month['day'] = [d.day for d in group_month['date']]  #group each year by month 
                grouped_by_day = group_month.groupby(group_month['day'])
                for name_day,_ in grouped_by_day:
                    days.append(name_day)
            missing_months = list(set(list(range(1,13,1))) - set(months))
            d = pd.DataFrame([[subj,name,len(months),missing_months,group.shape[0],len(days)]], columns = columns2)
            complete_data_info = pd.concat([d,complete_data_info])
        

    data_info = data_info.sort_values(by=[columns[0],columns[1]]) #sort by subject and file name
    missing_data_info = missing_data_info.sort_values(by=[columns[0],columns[1]]) #sort by subject and file name
    filename = '%s/sandbox/hhameed/data_info/' %root_directory #save in /data/PHI/PHI_OAPS/sandbox/hhameed/data_info/
    
    #uncomment to replace files
    
    data_info.to_csv(filename+'data_info.csv') 
    missing_data_info.to_csv(filename+'missing_data_info.csv') 
    complete_data_info.to_csv(filename+'complete_data_info.csv')


#plot CGM values for each for all the subjects
def plot_data():
    for subj in subjs:
        print(subj)
        data_frames,file_names = get_subject_data(subj, 'entries',oaps_data)
        for df, file_name in zip(data_frames,file_names):
            grouped = group_data(df,'year') #grouped according to years
            n = grouped.ngroups
            file_name = file_name.split('.')[0]
            i = 0
            dates = list()
            if n == 1:
                fig = plt.figure(figsize=(15,6))
                for name, group in grouped:
                    plt.plot(group['sgv'])
                    dates.append(name)
                    plt.ylabel('CGM values')
                    plt.xlabel('Time (5 min interval)')
                    plt.ylim(bottom=-1, top=500)
                    plt.grid()
            else:
                fig, axs = plt.subplots(n,1,figsize=(15,6))
                for name, group in grouped:
                    axs[i].plot(group['sgv'])
                    dates.append(name)
                    axs[i].set_ylabel('CGM values')
                    axs[i].set_xlabel('Time (min)')
                    axs[i].set_title(name)
                    axs[i].grid()
                    axs[i].set_ylim(bottom=-100, top=500)
                    i = i + 1
            t = 'Subject: '+subj
            plt.suptitle(t)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            filename = '%ssandbox/hhameed/plots/%s_%s.png' %(root_directory,subj,file_name)
            
            #uncomment to replace files
            plt.savefig(filename)

def inconsistent_values():
    columns=['Subject','Unique low values','no. of low values','%% of low values']
    data_info = pd.DataFrame(columns=columns)
    for subj in subjs:
        print('-----------------'+subj+'-------------------')
        _,data_directory,_ = get_files(oaps_data)
        subj_data_file = '%s%s/direct-sharing-31/extracted_files/complete_data_%s.csv' %(data_directory,subj,subj)
        if path.exists(subj_data_file) == True:
            df = pd.read_csv(subj_data_file, squeeze=True, index_col = 0)
            df = df.dropna(subset=['sgv'])

        new_df = df[df['sgv']<15]
        if not new_df.empty:
            low_values = np.unique(new_df['sgv'].values)
            percentage = round(100*(float(new_df.shape[0])/df.shape[0]),4)
  
            d = pd.DataFrame([[subj,low_values,new_df.shape[0],percentage]],columns=columns)
            data_info = pd.concat([d,data_info],ignore_index=True)

    filename = '%s/sandbox/hhameed/data_info/' %root_directory #save in /data/PHI/PHI_OAPS/sandbox/hhameed/data_info/ 
    data_info.sort_values(by='no. of low values',ascending=False,inplace=True)
    #uncomment to replace files 
    #data_info.to_csv(filename+'low_values.csv') 

if __name__ == "__main__":
    #get_subjects_with_predicted_BGS()
    #data_statistics()
    #compare_data_sets()
    #plot_data()
    #no_of_months_present()
    #inconsistent_values()
