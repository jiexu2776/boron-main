import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as stt
from scipy import stats
from scipy.optimize import curve_fit
#from numpy import arange

st.title('hello')
uploaded_files = st.file_uploader('upload files', type=['.exp'], accept_multiple_files=True)

#st.session_state.fNames = []
#fData = []

#for uploaded_file in st.session_state.uploaded_files:
#    bytes_data = uploaded_file.read()
#    st.session_state.fNames.append(uploaded_file.name)
#    fData.append(bytes_data)

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


def parseBoronTable(file):
    #content = file.read()
    content = file.getvalue().decode("utf-8")
    fname = file.__dict__["name"]
    _start = content.find("Cycle\tTime")
    #print(_start)

    _end = content.find("***\tCup")
    #print(_end)

    myTable = content[_start:_end-1]
    #print(myTable)

    cleanFname =f"temp/{fname}_cleanTable"
    with open(cleanFname ,"w") as _:
        _.write(myTable)

    df = pd.read_csv(cleanFname, 
                     sep='\t', 
                     #dtype="float"   #not working -->time
                     )

    return df, fname


def bacground_sub(select_line, factorSD, factor_B11):
    #fNames_tmp = sorted(st.session_state.fNames)
    average_B = []
    for i in st.session_state.uploaded_files:
        #read data
        #filename = fNames_tmp[i]
        #df = pd.read_csv(uploaded_files[i].read())
        df_data, filename = parseBoronTable(i)

        #print(df)
        #read all data rows and select useful columns
    #     fil = df['10B'].str.contains('L|IC|C|H') == True
    #     endnum = df['10B'][fil].index[0]
        df_data = df_data[['9.9', '10B', '10.2', '11B']].astype(float)
         #seperate two dataframe based on selectline, one is background, one is signal 
        index_select = df_data['10B'] >= select_line
        df_bacground_mean = df_data[~index_select].mean()
        df_signal = df_data.loc[index_select]
    #         #substract background, substract bulc for 10B and 11B
        df_bacground_sub = df_signal - df_bacground_mean
        df_bacground_sub['10B_bulc_sub'] = df_bacground_sub['10B']-(df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B_bulc_sub'] = df_bacground_sub['11B']-factor_B11*(df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B/10B'] = df_bacground_sub['11B_bulc_sub']/df_bacground_sub['10B_bulc_sub'] 

        res_iso = outlierCorrection(df_bacground_sub['11B/10B'], factorSD)
        res_11B = outlierCorrection(df_bacground_sub['11B'], factorSD)
        average_B.append({'filename': filename, '11B': np.mean(res_11B), '11B/10B_row': np.mean(res_iso), 'se': np.std(res_iso)/np.sqrt(len(res_iso))})
    
    df = pd.DataFrame(average_B)
    st.session_state.average_B = df
    

    return df
    
    #return df

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
    x_use = np.array(x)
    popt, pcov = curve_fit(polynomFit, xdata=x_use, ydata=y , p0=[0]*(order+1))
    fitData=polynomFit(x_use,*popt)
    
    res = []
    for unknown in listname:
        y_unknown = ref_stand / polynomFit(unknown,*popt)
        res.append({'factor': y_unknown})
    return(pd.DataFrame(res))

def regression_plot(x, y, ref_stand, order, listname):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='measuered', marker='o', linestyle='none' )
    x_use = np.array(x)
    popt, pcov = curve_fit(polynomFit, xdata=x_use, ydata=y , p0=[0]*(order+1))
    fitData=polynomFit(x_use,*popt)
    ax.plot(x_use, fitData, label='polyn. fit, order '+str(order), linestyle='--' )
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    return fig
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

#background substraction
if len(uploaded_files) != 0 :
    st.session_state.uploaded_files=uploaded_files
    df_data = bacground_sub(0.01, 1.5, 1)


#if "average_B" in st.session_state:
 #   st.session_state.average_B = df


#Choose A/B/C/D/U to get the regression for drift correction
sample_correction = 'A'

fil = df_data['filename'].str.contains(sample_correction)
df_data_B = df_data[fil]
df_data[' Sequence Number'] = selSmpType(df_data['filename'])

y_isotope = df_data_B['11B/10B_row']
y_11B = df_data_B['11B']
x = df_data_B.index.to_numpy()
#st.write(x)
#get the regression function and get all corrected factors for all measurements
factor_iso = regression(x,y_isotope, 4.05, 4, df_data.index.to_numpy())
factor_B = regression(x,y_11B, 35, 4, df_data.index.to_numpy())


#use corrected factors to correct machine drift and calculate isotope values for results
df_data['factor_iso'] = factor_iso
df_data['factor_B'] = factor_B

df_data['11B/10B_corrected'] = df_data['factor_iso']*df_data['11B/10B_row']
df_data['δ11B'] = ((df_data['11B/10B_corrected']/4.055)-1)*1000
df_data['δ11B_se'] = (df_data['se']*df_data['factor_iso']/4.055)*1000


#reorganize the laser file since many rows are wrong and useless
laser_file = st.file_uploader("Choose a CSV file", type = 'csv')
df_Laser = pd.read_csv(laser_file.name)

df_Laser_part1 = df_Laser[df_Laser[' Laser State'] == 'On'].iloc[:, [13, 20, 21]]
df_Laser_part2 = df_Laser[df_Laser[' Sequence Number'].notnull()].iloc[:, [1, 4]]

df_Laser_res = pd.concat([df_Laser_part2.reset_index(drop=True), df_Laser_part1.reset_index(drop=True)], axis=1)
# #merge laser data and neptune data
df_map1 = df_Laser_res.merge(df_data,on=' Sequence Number')




#sample_correction
ref = ((df_map1[df_map1['filename'].str.contains(sample_correction)][' Spot Size (um)']/2)**2).mean()
# #define the depth ratio
depth_ref = 30
depth_sample = 30
depth_ratio = depth_sample/depth_ref
# #use spot diameter, depth and signal intensity to calculate [B]
df_map1['[B]_corrected'] =  df_map1['11B']*df_map1['factor_B'] * (ref / ((df_map1[' Spot Size (um)']/2)**2) / depth_ratio)

trace = st.file_uploader("Choose a file", type = 'csv')
trace_file = pd.read_csv(trace.name)

#trace_file = pd.read_csv('2022-11-28-Si corrected-B5.csv') 

df_trace = prepare_trace(trace_file)

df_all = df_map1.merge(df_trace,on=' Sequence Number')
#df_all.to_csv('final.csv')
st.write(df_all)
result_csv = df_all.to_csv().encode('utf-8')
st.download_button(
    label = 'download results as .csv',
    data = result_csv,
    file_name = 'boron results.csv',
    mime = 'txt/csv',
)

st.pyplot(regression_plot(x,y_isotope, 4.05, 4, df_data.index.to_numpy()))