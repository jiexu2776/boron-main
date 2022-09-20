#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from curses import qiflush
from email.utils import collapse_rfc2231_value
from tty import CFLAG
import streamlit as st
import os
import pandas as pd
#from st_aggrid import AgGrid
import seaborn as sns
import pydeck as pdk
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FixedTicker
from random import randrange


# =============================================================================
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
# =============================================================================


st.session_state.tectSettingsPath = '/Users/dominik/Documents/GitHub/GeoROC/data/'
st.session_state.tectSettingsPath = 'data/'

st.session_state.el = pd.read_csv(st.session_state.tectSettingsPath + 'Archean Cratons/Bastar Craton.csv').columns[27:160]

st.session_state.tectSettingsFolder = os.listdir(st.session_state.tectSettingsPath)
st.session_state.tectSettings=[]
for i in st.session_state.tectSettingsFolder:
    if os.path.isdir(st.session_state.tectSettingsPath + i):
        st.session_state.tectSettings.append(i)

#---------------------------------#
#------ Welcome  -----------------#
#---------------------------------#  
def welcome():
    st.header('Welcome to the ✨Boron✨ world')
    st.write("Boron isotopes data processing starts here😜")
    
    

#---------------------------------#
#------ Scatter Plots  -----------#
#---------------------------------#  
 
#---------------------------------#
#------ Paired Plots  ------------#
#---------------------------------#  
    #import seaborn as sns

    

#---------------------------------#
#------ REE  ---------------------#
#---------------------------------#  


#---------------------------------#
#------ d  --------------------#
#---------------------------------#

#---------------------------------#
#------ Test  --------------------#
#---------------------------------#
def test():
    from bokeh.layouts import column, row
    from bokeh.models import CustomJS, Select
    from bokeh.plotting import ColumnDataSource, figure
    import numpy as np
    import altair as alt
    import pandas as pd
    from vega_datasets import data


    dfData = pd.read_csv('data/2022-02-15 Boron IC-F FF AAJ_5RMs_Seq2_20220215-221507/327_U.csv', encoding='utf-8', header=22)
    

    
    chart_data = dfData['Time'][1:200]
    chart_valu = dfData['10B'][1:200]
    #st.line_chart(dfData['10B'])

   # chart_data = pd.DataFrame(chart_data, chart_valu)

    dfData.iloc[:, 0:3][1:200]


# We use @st.experimental_memo to keep the dataset in cache
#    @st.experimental_memo
 #   def get_data():
  #      source = data.stocks()
   #     source = source[source.date.gt("2004-01-01")]
    #    return source

    #source = get_data()


    def get_data():
        source = data.stocks()
        source = source[source.date.gt("2004-01-01")]
        return source
    # Define the base time-series chart.
    def get_chart(data):
        hover = alt.selection_single(
            fields=["date"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="date",
            y="price",
            color="symbol",
        )
        )

    # Draw points on the line, and highlight based on selection

    points = lines.transform_filter(hover).mark_circle(size=65)
    # Draw a ÷rul?
            alt.Chart(data)
            .mark_rule()
            .encode(
`               x="yearmonthdate(date)",
                y="price",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("date", title="Date"),
                    alt.Tooltip("price", title="Price (USD)"),
                ],
            )
            .add_selection(hover)
        )
    return (lines + points + tooltips).interactive()

    chart = get_chart(source)

















        #source = pd.DataFrame(dfData.iloc[:, 1:3][1:200],
         #               columns=['Cycle', 'Time', '9.9'], index = dfData['Time'][1:200])

    
#---------------------------------#
#------ Main Page Sidebar --------#
#---------------------------------#  

st.sidebar.image('https://raw.githubusercontent.com/Hezel2000/GeoROC/main/images/Goethe-Logo.jpg', width=150)

page_names_to_funcs = {
    'Welcome': welcome,
   # 'Data Reading': scatterplots,
    #'Baseline Selection': paired_plots,
    #'Correction': REE,
    #'plot results': d,
    'test': test

}

demo_name = st.sidebar.radio("Select your Visualisation", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


link = '[Back to Geoplatform](http://www.geoplatform.de)'
st.sidebar.markdown(link, unsafe_allow_html=True)