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



def Process_test():
    st.session_state.tectSettingsPath = 'data/data to test/1. data folder20221129-214242'
    st.session_state.tectSettingsFolder = os.listdir(st.session_state.tectSettingsPath)



def sig_selection():

    average_B = []

    df_data, filename = parseBoronTable(st.session_state.sample_plot)
    df_data = df_data[['Cycle', '9.9', '10B', '10.2', '11B']].astype(float)

    fig, ax = plt.subplots()
    ax.plot(df_data['11B'], label='11B', c='green')
    ax.plot(df_data['10B'], label='10B', c='firebrick')
    ax.set_ylabel('signal intensity')
    ax.set_xlabel('cycle')
    x = df_data['11B'].index.to_numpy()
    ax.fill_between(x, max(df_data['11B']), where=(
        x < st.session_state.sig_end) & (x > st.session_state.sig_str), alpha=0.5)
    ax.fill_between(x, max(df_data['11B']), where=(
        x < st.session_state.bac_end) & (x > st.session_state.bac_str), alpha=0.5)

    ax.legend()
    return fig

st.title("""Please upload your data here""")
st.subheader('1. Upload your files from Neptune')

if st.button('Try test data here'):
    Process_test()
    st.session_state.stage_number = 1
    st.session_state.uploaded_files = []
    for file in st.session_state.tectSettingsFolder:
        if file.endswith('.exp'):
            st.session_state.uploaded_files.append(st.session_state.tectSettingsPath + '/' + file)

if st.button('Clear uploaded data'):
    st.session_state.uploaded_files = []

# len(st.session_state.uploaded_files) != 0:
if 'uploaded_files' in st.session_state and len(st.session_state.uploaded_files) != 0:
    uploaded_files = st.session_state.uploaded_files


else:
    st.session_state.uploaded_files = st.file_uploader('upload files', type=['exp'], accept_multiple_files=True)




if len(st.session_state.uploaded_files) != 0:

    st.session_state.sample_plot = st.selectbox(
        'Which is your sample to plot?',
        (st.session_state.uploaded_files))


    st.session_state.bac_str, st.session_state.bac_end = st.slider('Select background', 0, 200, (5, 70))
    st.session_state.sig_str, st.session_state.sig_end = st.slider('Select signal', 0, 200, (95, 175))
    
    st.pyplot(sig_selection())


#if "df_data" in st.session_state:
st.subheader('2. Upload your log file from Laser')
if st.button('Test laser data'):
    st.session_state.uploaded_laser_file = 'data/data to test/2. laser file.csv'

else:    
    st.session_state.uploaded_laser_file = st.file_uploader("Choose a laser file", type='csv')


