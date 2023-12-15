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



st.title("""Please upload your data here""")

def Process_test():
    st.session_state.tectSettingsPath = 'data/data to test/1. data folder20221129-214242'
    st.session_state.tectSettingsFolder = os.listdir(st.session_state.tectSettingsPath)
#st.write(pd.read_csv('data/data to test/1. data folder20221129-214242/001_A.exp', sep='\t'))






if st.button('Try test data here'):
    Process_test()
    st.session_state.stage_number = 1
    st.session_state.uploaded_files = []
    for file in st.session_state.tectSettingsFolder:
        if file.endswith('.exp'):
            # df = pd.read_csv(st.session_state.tectSettingsPath + '/' + file, sep='\t')
            # st.session_state.uploaded_files.append(df)
            st.session_state.uploaded_files.append(st.session_state.tectSettingsPath + '/' + file)


button_style = """
        <style>
        .stButton > button {
            color: black;
            background: lightblue;
            width: 200px;
            height: 50px;
        }

        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)



if st.button('Clear uploaded data'):
    st.session_state.uploaded_files = []

# len(st.session_state.uploaded_files) != 0:
if 'uploaded_files' in st.session_state and len(st.session_state.uploaded_files) != 0:
    uploaded_files = st.session_state.uploaded_files


else:
    st.session_state.uploaded_files = st.file_uploader('upload files', type=['exp'], accept_multiple_files=True)




st.session_state.sample_plot = st.selectbox(
    'Which is your sample to plot?',
    (st.session_state.uploaded_files))

def sig_selection():

    #fNames_tmp = sorted(st.session_state.fNames)
    average_B = []
    # if st.session_state.stage_number =  1:
    #     df_data, filename = Process_test(i)
    # else:
    #     df_data, filename = parseBoronTable(i)
    df_data, filename = parseBoronTable(st.session_state.sample_plot)
    df_data = df_data[['Cycle', '9.9', '10B', '10.2', '11B']].astype(float)

    fig, ax = plt.subplots()
    ax.plot(df_data['11B'], label='11B', c='green')
    ax.plot(df_data['10B'], label='10B', c='firebrick')
    ax.set_ylabel('signal intensity')
    ax.set_xlabel('cycle')
    #ax.axvline(x=select_index, color="red", linestyle="--")
    x = df_data['11B'].index.to_numpy()
    ax.fill_between(x, max(df_data['11B']), where=(
        x < st.session_state.sig_end) & (x > st.session_state.sig_str), alpha=0.5)
    ax.fill_between(x, max(df_data['11B']), where=(
        x < st.session_state.bac_end) & (x > st.session_state.bac_str), alpha=0.5)

    ax.legend()
    return fig

st.session_state.bac_str, st.session_state.bac_end = st.slider('Select background', 0, 200, (5, 70))
st.session_state.sig_str, st.session_state.sig_end = st.slider('Select signal', 0, 200, (95, 175))
sig_selection()
