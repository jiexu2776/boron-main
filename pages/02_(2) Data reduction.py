import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as stt
from scipy import stats
from scipy.optimize import curve_fit
import os
import re
from io import StringIO


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/website-profile.gif);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 100px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Main ";
                margin-left: 100px;
                margin-top: 10px;
                font-size: 25px;
                position: relative;
                top: 100px;

            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()



st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.image(
    'https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/Goethe-Logo.gif')



st.header('B-Isotopes data reduction')


def parseBoronTable(file):
    if isinstance(file, str):
        with open(file, "r") as _:
            content= _.read()
        fname  = file.split('/')[-1]
    else:   # streamlit file object
        content = file.getvalue().decode("utf-8")
        fname = file.__dict__["name"]
    _start = content.find("Cycle\tTime")
    _end = content.find("***\tCup")
    myTable = content[_start:_end-1]

    df = pd.read_csv(StringIO(myTable),
                     sep='\t',
                     # dtype="float"   #not working -->time
                     )
    return df, fname



def outlierCorrection(data, factorSD):
    element_signal = np.array(data)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)
    fil = (data < mean + factorSD * sd) & (data > mean - factorSD * sd)
    res=data[fil]
    outlier=data[~fil]
    return res, outlier


def selSmpType(dataFiles):
    re_patter = r"(\d{3}\_[a-zA-Z])"


    l = []
    for file in dataFiles:
        # print(file)
        match = re.search(re_patter,file)[0]
        print(match)
        l.append(match)
        #if "/" 
        #'data/data to test/1. data folder20221129-214242/file.....'
        #l.append(float(file.split("/")[-1].split('_')[0]))
    return l





def bacground_sub(factorSD):
    average_B = []
    for i in st.session_state.uploaded_files:

        df_data, filename = parseBoronTable(i)
        df_data = df_data[['Cycle', '9.9','10B', '10.2',  '10.627', '10.9' ,'11B']].astype(float)


        df_bacground_mean = df_data[st.session_state.bac_str:st.session_state.bac_end].mean()
        df_signal = df_data[st.session_state.sig_str:st.session_state.sig_end]

    #         #substract background, substract bulc for 10B and 11B
        df_bacground_sub = df_signal - df_bacground_mean

        df_bacground_sub['10B_bulc_sub'] = df_bacground_sub['10B'] - (df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B_bulc_sub'] = df_bacground_sub['11B'] - (df_bacground_sub['10.627']+df_bacground_sub['10.9'])/2
        df_bacground_sub['11B/10B'] = df_bacground_sub['11B_bulc_sub'] / df_bacground_sub['10B_bulc_sub']
        

        res_iso, res_iso_outlier = outlierCorrection(df_bacground_sub['11B/10B'], factorSD)



        res_11B, res_11B_outlier = outlierCorrection(df_bacground_sub['11B'], factorSD)


        if i == st.session_state.sample_plot:
            fig1, ax = plt.subplots()
            ax.plot(df_bacground_sub['11B/10B'], 'ko')
            ax.plot(res_iso_outlier, 'ro', label='outliers')
            ax.set_ylabel('$^{11}B$/$^{1O}B$')
            ax.legend()
            ax.axhline(y=res_iso.mean(), color="black", linestyle="--")

            st.pyplot(fig1)
        #   
        average_B.append({'filename': filename, '11B': np.mean(res_11B), '11B/10B_row': np.mean(res_iso), 'se': np.std(res_iso)/np.sqrt(len(res_iso))})

    df = pd.DataFrame(average_B)
    st.session_state.average_B = df
    st.session_state.fig1=fig1

    return df, fig1

def polynomFit(inp, *args):
    x = inp
    res = 0
    for order in range(len(args)):
        res += args[order] * x**order
    return res


def regression(x, y, ref_stand, order, listname):
    #order = st.session_state.regress
    fig2, ax = plt.subplots()
    ax.plot(x, y, label='measuered', marker='o', linestyle='none')
    # x_use = np.array(x)
    popt, pcov = curve_fit(polynomFit, xdata=x,
                           ydata=y, 
                           p0=[0]*(int(order)+1)
                           )
    fitData = polynomFit(x, *popt)

    ax.plot(x, fitData, label='polyn. fit, order ' +
            str(order), linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylabel('raw data')
    ax.set_xlabel('sequence')
    st.pyplot(fig2)
    res = []
    for unknown in listname:
        y_unknown = ref_stand / polynomFit(unknown, *popt)
        res.append({'factor': y_unknown})
    return(pd.DataFrame(res))






st.subheader('Set outlier')
st.write('outlier factor: means data is outlier_factor times of sd will be cut')
    


outlier_factor = st.number_input('outlier factor', value=1.5)


if "average_B" in st.session_state:
    #A  = st.info("Reloading already parsed dataframe!")
    st.session_state.df_data = st.session_state.average_B
else:
    st.session_state.df_data = bacground_sub(outlier_factor)


st.subheader('Drift correction')
st.write('Please choose your standard for boron isotopes correction')

col1, col2 = st.columns([1, 3])

with col1:
    standard = st.selectbox(
        'Select standard',
        ('GSD-1G', 'NIST SRM 612', 'B5', 'GSC-1G'))
    if standard == 'B5':
        st.session_state.number_iso = float(4.0332057)
        st.session_state.number_trace = float(8.42)
        st.session_state.SRM951_value = float(4.0492)

    if standard == 'NIST SRM 612':
        st.session_state.number_iso = float(4.05015)
        st.session_state.number_trace = float(35)
        st.session_state.SRM951_value = float(4.0545)

    if standard == 'GSD-1G':
        st.session_state.number_iso = float(4.09548)
        st.session_state.number_trace = float(50)
        st.session_state.SRM951_value = float(4.0545)

    if standard == 'GSC-1G':
        st.session_state.number_iso = float(4.1378)
        st.session_state.number_trace = float(22)
        st.session_state.SRM951_value = float(4.04362)

    st.session_state.standard_values = {
        "number_iso" : st.session_state.number_iso,
        "number_trace" : st.session_state.number_trace,
        "SRM951_value" : st.session_state.SRM951_value

    }

    st.write(st.session_state.standard_values)
    st.session_state.sample_correction = st.selectbox(
        'Select standard designation',
        ('A', 'B', 'C', 'D'))


    st.session_state.default_reg_level = 4
    st.session_state.regress_level = st.number_input('regression level (4 is recommended)', step=1, value=st.session_state.default_reg_level, format='%X'
                                                        )

# Choose A/B/C/D/U to get the regression for drift correction



with col2:
    st.session_state.df_data['file name'] = selSmpType(st.session_state.df_data['filename'])
    # print(st.session_state.df_data['file name'])


    s = []
    for i in st.session_state.df_data['file name']:
        s.append(int(i.split('_')[0]))
    st.session_state.df_data[' Sequence Number'] = s

    st.session_state.df_data.sort_values(by = [' Sequence Number'], inplace=True)#.reset_index(drop = True)

    fil = st.session_state.df_data['file name'].str.contains(st.session_state.sample_correction)
    st.session_state.df_data_B = st.session_state.df_data[fil]

    y_isotope = st.session_state.df_data_B['11B/10B_row'].astype(float)
    y_11B = st.session_state.df_data_B['11B'].astype(float)
    x = st.session_state.df_data_B[' Sequence Number']
    # x = df_data_B.index.to_numpy()
    # st.write(x)
    # get the regression function and get all corrected factors for all measurements
    #factor_iso = regression(x,y_isotope, 4.05, 4, df_data.index.to_numpy())
    #factor_B = regression(x,y_11B, 35, 4, df_data.index.to_numpy())
    factor_iso = regression(x, y_isotope,st.session_state.number_iso,
                            st.session_state.regress_level if "regress_level" in st.session_state else st.session_state.default_reg_level,
                            st.session_state.df_data[' Sequence Number'])
                            # df_data.index.to_numpy()
                            

    # st.write(x)
    # get the regression function and get all corrected factors for all measurements

# use corrected factors to correct machine drift and calculate isotope values for results
st.session_state.df_data['factor_iso'] = factor_iso

st.session_state.df_data['11B/10B_corrected'] = st.session_state.df_data['factor_iso']*st.session_state.df_data['11B/10B_row']
st.session_state.df_data['δ11B'] = ((st.session_state.df_data['11B/10B_corrected']/st.session_state.SRM951_value)-1)*1000
st.session_state.df_data['δ11B_se'] = (st.session_state.df_data['se']*st.session_state.df_data['factor_iso']/st.session_state.SRM951_value)*1000

##df_data_B is a dataframe for standard, df_data is a dataframe for all samples;
st.session_state.df_data = st.session_state.df_data
st.session_state.df_data_B = st.session_state.df_data_B


if st.session_state.uploaded_laser_file is not None:
        st.session_state.df_Laser = pd.read_csv(st.session_state.uploaded_laser_file)

        st.session_state.df_Laser_part1 = st.session_state.df_Laser[st.session_state.df_Laser[' Laser State']
                                == 'On'].iloc[:, [13, 20, 21]]
        st.session_state.df_Laser_part2 = st.session_state.df_Laser[st.session_state.df_Laser[' Sequence Number'].notnull()].iloc[:, [
                1, 4]]

        st.session_state.df_Laser_res = pd.concat([st.session_state.df_Laser_part2.reset_index(
                drop=True), st.session_state.df_Laser_part1.reset_index(drop=True)], axis=1)

                
            # #merge laser data and neptune data

        st.session_state.df_map1 = st.session_state.df_Laser_res.merge(st.session_state.df_data, on=' Sequence Number')
        

        st.subheader('2.1 B concerntration correction')

        #st.session_state.default_reg_level_B = 4
        st.session_state.regress_level_B = st.number_input('insert your regression level for [B] (4 is recommended)', 
        step=1, 
        value=st.session_state.default_reg_level, 
        format='%X'
                                                        )     


        y_isotope = st.session_state.df_data_B['11B/10B_row']
        y_11B = st.session_state.df_data_B['11B']

        x = st.session_state.df_data_B[' Sequence Number']
        factor_B = regression(x, y_11B, st.session_state.standard_values["number_trace"],
                        st.session_state.regress_level_B if "regress_level_B" in st.session_state else st.session_state.default_reg_level_B, 
                        st.session_state.df_data[' Sequence Number']
                        )
        st.session_state.df_map1['factor_B'] = factor_B
        

        depth_ref = st.number_input('insert the abalation depth of selected reference / µm', value = 30.0)
        depth_sample = st.number_input('insert the abalation depth of other samples / µm', value = 30.0)
                
        depth_ratios = []
        for i in st.session_state.df_map1['file name'].str.contains('A'):
            if i == True:
                depth_ratio = 1 
            else:
                depth_ratio = depth_sample / depth_ref
            depth_ratios.append(depth_ratio)

        st.session_state.df_map1['depth_correction'] = depth_ratios

        spot_shape = st.selectbox(
                    'What is the type of your spots?',
                    ('circle', 'squre'))
        if spot_shape == 'circle':
            st.session_state.df_map1[' Spot Size (um)'] = st.session_state.df_Laser_res[' Spot Size (um)']
            ref = ((st.session_state.df_map1[st.session_state.df_map1['file name'].str.contains(st.session_state.sample_correction)][' Spot Size (um)']/2)**2).mean()
            st.session_state.df_map1['[B]_corrected'] = st.session_state.df_map1['11B']*st.session_state.df_map1['factor_B'] * (ref / ((st.session_state.df_map1[' Spot Size (um)']/2)**2) / depth_ratios)

        if spot_shape == 'squre':

            dia = st.session_state.df_map1[' Spot Size (um)']
            spotsize = dia.str.split(' ').str[0].apply(lambda x: float(x))
            st.session_state.df_map1[' Spot Size (um)'] = spotsize
            ref = ((st.session_state.df_map1[st.session_state.df_map1['file name'].str.contains(st.session_state.sample_correction)][' Spot Size (um)'])**2).mean()
            st.session_state.df_map1['[B]_corrected'] = st.session_state.df_map1['11B']*st.session_state.df_map1['factor_B'] * (ref / ((st.session_state.df_map1[' Spot Size (um)'])**2) / depth_ratios)   

        st.session_state.df_map1 = st.session_state.df_map1