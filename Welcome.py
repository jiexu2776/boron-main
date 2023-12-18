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
                background-image: url('https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/website-profile.gif');
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 100px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Main";
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
    'images/Goethe-Logo.gif')




st.title("""Hello, welcome to the boron world""")

st.header('Here are five pages from left Main menu', divider='rainbow')

st.subheader(':red[(1) Import data:] upload datafiles from Neptune and laser')

st.subheader(':green[(2) Data reduction:] choose standard for intra-sequence instrumental correction')


st.subheader(':blue[(3)Data virualization and download:] check results and download files')
