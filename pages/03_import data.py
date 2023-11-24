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


def find_exp_filenames( path_to_dir, suffix=".exp" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]







st.title("""Hello, welcome to the boron world""")



st.header('1 Please upload your data files from Neptune')


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
