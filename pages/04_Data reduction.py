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


def outlierCorrection(data, factorSD):
    element_signal = np.array(data)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)
    fil = (data < mean + factorSD * sd) & (data > mean - factorSD * sd)
    res=data[fil]
    outlier=data[~fil]
    return res, outlier

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
            st.pyplot(fig1)
        #   
        average_B.append({'filename': filename, '11B': np.mean(res_11B), '11B/10B_row': np.mean(res_iso), 'se': np.std(res_iso)/np.sqrt(len(res_iso))})

    df = pd.DataFrame(average_B)
    st.session_state.average_B = df
    st.session_state.fig1=fig1

    return df, fig1

st.subheader('Set outlier')
st.write('outlier factor: means data is outlier_factor times of sd will be cut')
    
outlier_factor = st.number_input('outlier factor', value=1.5)




#if "average_B" in st.session_state:
    #A  = st.info("Reloading already parsed dataframe!")
 #   df_data = st.session_state.average_B
#else:
df_data, figure1 = bacground_sub(outlier_factor)

