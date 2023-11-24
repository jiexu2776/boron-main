#!/usr/bin/env python
# coding: utf-8

# <h1> B Data Reduction </h1>

# # Importing libraries

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
import warnings
import statistics as stt
import os
from scipy import stats
from scipy.optimize import curve_fit
from numpy import arange
from collections import deque
import string
import math


# # Functions

# In[73]:


#-------------------------------
# Loading & select different type of data, z.b B, C, U
#-------------------------------

def dirFiles(fDir, ending):
    fNames = []
    for i in os.listdir(fDir):
        if i.endswith(ending):
            fNames.append(i)
    return fNames

#-------------------------------
# used for mapping
#-------------------------------

def selSmpType(dataFiles):
    l = []
    for file in dataFiles:    
        l.append(float(file.split('_')[0]))
    return l

#-------------------------------
# Outlier Correction and Data Conversion from str to float
#-------------------------------
def outlierCorrection(data, factorSD):
    element_signal = np.array(data)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)
    return [x for x in data if (x > mean - factorSD * sd) and (x < mean + factorSD * sd)]

#-------------------------------
# background substact for 10B with the average of 9.9 and 10.2, think abut each line has to be connected
#-------------------------------

def bacground_sub(folder, select_line, factorSD, factor_B11):
    listname = dirFiles('/Users/mila/Documents/GitHub/boron-main/data/' + folder, 'exp')
    listname.sort()
    average_B = []
    for filename in listname:
        #read data
        df = pd.read_csv('/Users/mila/Documents/GitHub/boron-main/data/' +folder + '/' +filename, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
        #read all data rows and select useful columns
        fil = df['10B'].str.contains('L|IC|C|H') == True
        endnum = df['10B'][fil].index[0]
        df_data = df[:endnum][['9.9', '10B', '10.2', '11B']].astype(float)
            #seperate two dataframe based on selectline, one is background, one is signal 
        index_select = df_data['10B'] >= select_line
        df_bacground_mean = df_data[~index_select].mean()
        df_signal = df_data.loc[index_select]
            #substract background, substract bulc for 10B and 11B
        df_bacground_sub = df_signal - df_bacground_mean
        df_bacground_sub['10B_bulc_sub'] = df_bacground_sub['10B']-(df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B_bulc_sub'] = df_bacground_sub['11B']-factor_B11*(df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B/10B'] = df_bacground_sub['11B_bulc_sub']/df_bacground_sub['10B_bulc_sub'] 

        res_iso = outlierCorrection(df_bacground_sub['11B/10B'], factorSD)
        res_11B = outlierCorrection(df_bacground_sub['11B'], factorSD)
        average_B.append({'filename': filename, '11B': np.mean(res_11B), '11B/10B_row': np.mean(res_iso), 'se': np.std(res)/math.sqrt(len(res))})
    return (pd.DataFrame(average_B))


#-------------------------------
# regression based on the level from 2-5 you chosed
#-------------------------------

def polynomFit(inp, *args):
    x=inp
    res=0
    for order in range(len(args)):
        res+=args[order] * x**order
    return res


def regression(x, y, ref_stand, order, listname):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='measuered', marker='o', linestyle='none' )
    x_use = np.array(x)
    popt, pcov = curve_fit(polynomFit, xdata=x_use, ydata=y , p0=[0]*(order+1))
    fitData=polynomFit(x_use,*popt)
    ax.plot(x_use, fitData, label='polyn. fit, order '+str(order), linestyle='--' )
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    res = []
    for unknown in listname:
        y_unknown = ref_stand / polynomFit(unknown,*popt)
        res.append({'factor': y_unknown})
    return(pd.DataFrame(res))


#-------------------------------
# data from machine for trace elements couldn't be used directly for plotting or calculation
## delete repete Ca44 and Mg26 data, unify the formate of column titles, and float all str data.
#-------------------------------

def prepare_trace(datafile):
    if 'LR' in datafile.columns[14]:
        del datafile['44Ca(LR)']
        del datafile['26Mg(LR)']
    else:
        del datafile['44Ca']
        del datafile['26Mg']
    
    datafile.columns = datafile.columns.str.replace('\d+', '')
    datafile.columns = datafile.columns.str.replace('\('+'LR'+'\)', '')
    res = []
    for i in range(13, len(datafile.columns)):
        for j in datafile.iloc[:, i]:
            if '<' in j:
                res.append(j)
    RES = datafile.replace(to_replace = res, value='nan', regex=True)
    RES2 = RES.replace({'ERROR: Error (#1002): Internal standard composition can not be 0': np.nan})
    RES3 = RES2.replace({'ERROR: Error (#1003): Calibration RM composition does not contain analyte element': np.nan})
    RES4 = RES3.iloc[:, 13:].astype(float)
    columns = RES3.iloc[:, 13:].columns
    RES4[columns] = RES3.iloc[:, 13:]
    RES4[' Sequence Number'] = RES3['LB#']
    return(RES4)


# # machine drift correction and isotope results calculations

# In[76]:


#background substraction
df_data = bacground_sub('2022-11-28 B Carb Sy MC JX AG splitstream_20221128-203622', 0.01, 1.5, 1)
#Choose A/B/C/D/U to get the regression for drift correction
sample_correction = 'A'

fil = df_data['filename'].str.contains(sample_correction)
df_data_B = df_data[fil]
df_data[' Sequence Number'] = selSmpType(df_data['filename'])

y_isotope = df_data_B['11B/10B_row']
y_11B = df_data_B['11B']
x = df_data_B.index.to_numpy()
#get the regression function and get all corrected factors for all measurements
factor_iso = regression(x,y_isotope, 4.05, 4, df_data.index.to_numpy())
factor_B = regression(x,y_11B, 35, 4, df_data.index.to_numpy())


#use corrected factors to correct machine drift and calculate isotope values for results
df_data['factor_iso'] = factor_iso
df_data['factor_B'] = factor_B

df_data['11B/10B_corrected'] = df_data['factor_iso']*df_data['11B/10B_row']
df_data['δ11B'] = ((df_data['11B/10B_corrected']/4.055)-1)*1000
df_data['δ11B_se'] = (df_data['se']*df_data['factor_iso']/4.055)*1000


# # Mapping: Laser parameters and isotope results

# In[115]:


#read laser file
df_Laser = pd.read_csv('data/seq1_log_20221128_203700.csv', header=0)
#reorganize the laser file since many rows are wrong and useless
df_Laser = pd.read_csv('data/seq1_log_20221128_203700.csv', header=0)
df_Laser_part1 = df_Laser[df_Laser[' Laser State'] == 'On'].iloc[:, [13, 20, 21]]
df_Laser_part2 = df_Laser[df_Laser[' Sequence Number'].notnull()].iloc[:, [1, 4]]

df_Laser_res = pd.concat([df_Laser_part2.reset_index(drop=True), df_Laser_part1.reset_index(drop=True)], axis=1)
#merge laser data and neptune data
df_map1 = df_Laser_res.merge(df_data,on=' Sequence Number')


# # Calculate B concentration from signal intensity of Neptune

# In[119]:


#sample_correction
ref = ((df_map1[df_map1['filename'].str.contains(sample_correction)][' Spot Size (um)']/2)**2).mean()
#define the depth ratio
depth_ref = 30
depth_sample = 30
depth_ratio = depth_sample/depth_ref
#use spot diameter, depth and signal intensity to calculate [B]
df_map1['[B]_corrected'] =  df_map1['11B']*df_map1['factor_B'] * (ref / ((df_map1[' Spot Size (um)']/2)**2) / depth_ratio)


# # Mapping:  isotope results and trace elements

# In[122]:


df_trace = prepare_trace(pd.read_csv('/Users/mila/Documents/博士论文/实验数据/trace elements/2022-11-28 TREE Hallmann Sy MC JX AG/results/code/2022-11-28-Si-NISTnnp.csv'))

df_all = df_map1.merge(df_trace,on=' Sequence Number')
#df_all.to_csv('final.csv')
df_all


# # Testing

# In[8]:


results = {'Laser Energy (mJ)': res_energy, 
           'Spot Size (um)': res_spotsize, 'Laser HV (kV)': res_HV}

Laser_res= pd.DataFrame(results)
resss = pd.concat([Laser_res, dfN.reindex(Laser_res.index)], axis=1)


# In[14]:


def laser_Cond(smp):
    return resss[resss['sequence'].str.contains(smp)]

laser_Cond('A')


# ### Some old code

# In[ ]:


#check for outliers for signal:
    element_signal = np.array(fils)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)
    final_list_signal = [x for x in fils if (x > mean - factorSD * sd)]
    final_list_signal = [x for x in final_list_signal if (x < mean + factorSD * sd)]
    
#check for outliers for background:
    element_backg = np.array(filw)
    mean = np.mean(element_backg, axis=0)
    sd = np.std(element_backg, axis=0)
    final_list_backg = [x for x in filw if (x > mean - factorSD * sd)]
    final_list_backg = [x for x in final_list_backg if (x < mean + factorSD * sd)]
    

def calculateSingleFile(filename, bg_10B, factorSD):
#upload data:
    df = pd.read_csv('data/2022-02-15 Boron measurements/' + filename, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
#calculate average signal and average backgroud:
    endnum = df['10B'][df['10B'] == 'L3'].index[0]
#    B_10 = []
    fils = []
    filw = []
#    for i in df['10B'][0:endnum]:
#        B_10.append(float(i)) #conversion to float required, as numbers are stored as str
#    for i in B_10:
    for i in df['10B'][0:endnum]:
        meas = float(i)
        if (meas > bg_10B) == True:
            fils.append(meas)
        else:
            filw.append(meas)

#    outlier_corrected_signal =outlierCorrection(fils, factorSD)
#    outlier_corrected_backg =outlierCorrection(filw, factorSD)

#    B_10_aver = sum(outlier_corrected_signal )/len(outlier_corrected_signal )
#    B_10_backg_aver = sum(outlier_corrected_backg)/len(outlier_corrected_backg)

    B_10_aver = np.mean(outlierCorrection(fils, factorSD))
    B_10_backg_aver = np.mean(outlierCorrection(filw, factorSD))
    
    return(B_10_aver, B_10_backg_aver)


def background_substract_basic(filename, signal_object, select_line, factorSD):
#upload data:
    df = pd.read_csv('data/2022-11-04 Boron measurements/' + filename, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
#calculate average signal and average backgroud:

    data_float = data_convert(df[signal_object]) #conversion to float required, as numbers are stored as str
    fils = []
    filw = []
        
    for i in data_float:
        if (i > select_line) == True:
            fils.append(i)
        else:
            filw.append(i)

    signal_aver = np.mean(outlierCorrection(fils, factorSD))
    backg_aver = np.mean(outlierCorrection(filw, factorSD))
    res = signal_aver - backg_aver
    return(res)


def background_substract_interference_bacground(filename, select_line, factorSD, factor_B11):
    
    df = pd.read_csv('data/2022-11-04 Boron measurements/' + filename, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
    res_bacg_inter_sub = []
    data_interference_left = data_convert(df['9.9'])
    data_interference_right = data_convert(df['10.2'])
    data_B10 = data_convert(df['10B'])
    data_B11 = data_convert(df['11B'])

    bacgrounds = []
    bacgrounds_10 = []
    bacgrounds_11 = []
    bacgrounds_99 = []
    bacgrounds_102 = []
    for i in data_B10:
        if (i < select_line) == True:
            index = data_B10.index(i)
            j = data_B11[index]
            bacgrounds_10.append(i)
            bacgrounds_11.append(j)
            bacgrounds_99.append(i)
            bacgrounds_102.append(j)
    backg10_aver = np.mean(outlierCorrection(bacgrounds_10, factorSD))        
    backg11_aver = np.mean(outlierCorrection(bacgrounds_11, factorSD))        
    backg99_aver = np.mean(outlierCorrection(bacgrounds_10, factorSD))
    backg102_aver = np.mean(outlierCorrection(bacgrounds_102, factorSD))

    for i in data_B10:
        if (i > select_line) == True:
            index = data_convert(df['10B']).index(i)
            interference_left_back_sub = data_interference_left[index] - backg99_aver
            interference_right_back_sub = data_interference_right[index] - backg102_aver
            interference = (interference_left_back_sub + interference_right_back_sub)/2
            res_10 = i - interference - backg10_aver
            res_11 = data_B11[index] - (interference * factor_B11) - backg11_aver
            #offset = res_11/res_10 - 
            res_bacg_inter_sub.append (res_11/res_10)

        res = outlierCorrection(res_bacg_inter_sub, factorSD)
    return np.mean(res)


#-------------------------------
#background substact with the basic method, signal_aver-background_aver
#-------------------------------

def background_substract_basic(filename, signal_object, select_line, factorSD):
#upload data:
    df = pd.read_csv('data/2022-11-04 Boron measurements/' + filename, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
#read all data rows and select useful columns
    fil = df['10B'].str.contains('L|IC|C|H') == True
    endnum = df['10B'][fil].index[0]
    df_data = df[:endnum][['10B', '11B']].astype(float)
#seperate two dataframe based on selectline, one is background, one is signal 
    index_select = df_data[signal_object] >= select_line
    df_bacground_mean = df_data[~filx].mean()
    df_signal = df_data.loc[index_select]
#substract background, substract bulc for 10B and 11B
    df_bacground_sub = df_signal - df_bacground.mean()                            
    res = outlierCorrection(df_bacground_sub['11B']/df_bacground_sub['10B'], factorSD)
    return(res)



dataFiles = dirFiles('data/2022-11-04 Boron measurements', 'exp')
dataFiles.sort()

dataFiles = selSmpType('B')

dataFiles

res = []
for i in dataFiles:
    aver = background_substract_basic(i, '11B', 0.005, 2)/background_substract_basic(i, '10B', 0.0005, 2)
    Neptune_res = {'sequence': i, 'average':aver}
    res.append(Neptune_res)

x_b = getfilenumber(pd.DataFrame(res)['sequence'], '_')

#def offset(datafile):
 #   res = []
  #  for i in datafile:
   #     res.append(objective(i, a, b, c, d, e, f))
    #return res

#offset(x_b)
res = []
for i in dataFiles:
    aver = background_substract_basic(i, '11B', 0.005, 2)/background_substract_basic(i, '10B', 0.0005, 2)
    Neptune_res = {'sequence': i, 'average':aver}
    res.append(Neptune_res)
res




def data_convert(data):
    data_float = []
    fil =  (data.str.contains('L|IC|C|H') == True)
    endnum = data[fil].index[0]
    for i in data[0:endnum]:
        data_float.append(float(i))
    return data_float




def selSmpType(dataFiles, smp):
    l = []
    for file in dataFiles:    
        if smp in file.split('_')[1]:
            l.append(file)
    return l

#-------------------------------
# read the number from Neptune filename
#-------------------------------

def getfilenumber(datafile, simbel):
    res = []
    for i in datafile:
        res.append(float(i.split(simbel)[0]))
    return res


#-------------------------------
# average valye calculation
#-------------------------------
def calculate_aver(dataFiles, IC):
    res = []
    for i in dataFiles:
        aver = background_substract_basic(i, '11B',  0.005, 2)/background_substract_basic(i, '10B', 0.0005, 2)
        Neptune_res = {'sequence': i, 'average':aver}
        res.append(Neptune_res)
        df = pd.DataFrame(res)
        
    return df

