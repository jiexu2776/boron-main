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



st.header('Split stream data')

    if "df_map1" in st.session_state:
        # st.divider()
        st.header('Results')
        st.write('Export results or append your trace elements')
    
        trace_file = st.radio(
            'Select',
            ('No Split Stream', 'Split Stream'), horizontal=True)
    
        if trace_file == 'No Split Stream':
            st.session_state.df_all = st.session_state.df_map1
        
            st.session_state.option = st.selectbox(
                'Which srandard data would you like to display?',
                ('A', 'B', 'C', 'D'))

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # ax.plot([0,1],[0,1], transform=ax.transAxes, c = 'red')
        filA = st.session_state.df_all['file name'].str.contains(st.session_state.option)
        # filB = st.session_state.df_all['file name'].str.contains('B')
        # filC = st.session_state.df_all['file name'].str.contains('C')
        # filD = st.session_state.df_all['file name'].str.contains('D')

        ax.errorbar(st.session_state.df_all[filA]['[B]_corrected'], st.session_state.df_all[filA]['δ11B'], yerr = st.session_state.df_all[filA]['δ11B_se'], c = 'green', fmt ='o', label  = st.session_state.df_all[filA][' Comment'].drop_duplicates().values[0])
        # axs[0, 1].errorbar(st.session_state.df_all[filB]['[B]_corrected'], st.session_state.df_all[filB]['δ11B'], yerr = st.session_state.df_all[filB]['δ11B_se'], c = 'brown', fmt ='o', label  = st.session_state.df_all[filB][' Comment'].drop_duplicates().values[0])

        # axs[1, 0].errorbar(st.session_state.df_all[filC]['[B]_corrected'], st.session_state.df_all[filC]['δ11B'], yerr = st.session_state.df_all[filC]['δ11B_se'], c = 'darkblue', fmt ='o', label  = st.session_state.df_all[filC][' Comment'].drop_duplicates().values[0])
        # axs[1, 1].errorbar(st.session_state.df_all[filD]['[B]_corrected'], st.session_state.df_all[filD]['δ11B'], yerr = st.session_state.df_all[filD]['δ11B_se'], c = 'darkorange', fmt ='o', label  = st.session_state.df_all[filD][' Comment'].drop_duplicates().values[0])
        #axs[0,0].set_ylabel('[B]_measured by Element')
        fig.text(0.5,0.04, "[B]_corrected by Neptune (μg/g)", ha="center", va="center")
        fig.text(0.05,0.5, '$\u03B4^{11}$B$_{}$ (‰)', ha="center", va="center", rotation=90)
        # axs[0,0].legend()
        # axs[0,1].legend()
        # axs[1,0].legend()
        ax.legend()
        #axs.xlabel('[B]_corrected by Neptune')
        st.pyplot(fig)
        fn = 'fig.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image for standards",
                data=img,
                file_name=fn,
                mime="image/png"
            )

    elif trace_file == 'Split Stream':
        st.header('3. Please upload your trace element data processed from Ladr')



        st.session_state.trace = st.file_uploader("Choose a file", type='csv', accept_multiple_files=True)
        if "trace" in  st.session_state and len(st.session_state.trace) > 0:

            trace_file = pd.read_csv(st.session_state.trace[0])

            #trace_file = pd.read_csv('2022-11-28-Si corrected-B5.csv')

            df_trace = prepare_trace(trace_file)
            # st.write(df_trace)

            st.session_state.df_all = st.session_state.df_map1.merge(df_trace, on=' Sequence Number')
            # fig4, ax = plt.subplots()
            # ax.plot([0,1],[0,1], transform=ax.transAxes, c = 'red')
            # ax.scatter(st.session_state.df_all['[B]_corrected'], st.session_state.df_all['B'], s =70, c = 'darkorange', edgecolors = 'black')
            # ax.set_ylabel('[B]_measured by Element')
            # ax.set_xlabel('[B]_corrected by Neptune')
            # st.pyplot(fig4)

            st.session_state.option = st.selectbox(
                'Which srandard data would you like to display?',
                ('A', 'B', 'C', 'D'))


            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
            # ax.plot([0,1],[0,1], transform=ax.transAxes, c = 'red')
            filA = st.session_state.df_all['file name'].str.contains(st.session_state.option)
            # st.write(st.session_state.df_all[filA])
            # filB = st.session_state.df_all['file name'].str.contains('B')
            # filC = st.session_state.df_all['file name'].str.contains('C')
            # filD = st.session_state.df_all['file name'].str.contains('D')
            # st.write(st.session_state.df_all[filA]['11B_y'])
            # st.write(st.session_state.df_all[filA]['δ11B'])
            # st.write(st.session_state.df_all[filA]['δ11B_se'])
            # ax.errorbar(st.session_state.df_all[filA]['B'], st.session_state.df_all[filA]['δ11B'], yerr = st.session_state.df_all[filA]['δ11B_se'])

            ax.errorbar(st.session_state.df_all[filA]['11B_y'], st.session_state.df_all[filA]['δ11B'], yerr = st.session_state.df_all[filA]['δ11B_se'], c = 'green', fmt ='o', label  = st.session_state.df_all[filA][' Comment'].drop_duplicates().values[0])
            # axs[0, 1].errorbar(st.session_state.df_all[filB]['B'], st.session_state.df_all[filB]['δ11B'], yerr = st.session_state.df_all[filB]['δ11B_se'], c = 'brown', fmt ='o', label  = st.session_state.df_all[filB][' Comment'].drop_duplicates().values[0])

            # axs[1, 0].errorbar(st.session_state.df_all[filC]['B'], st.session_state.df_all[filC]['δ11B'], yerr = st.session_state.df_all[filC]['δ11B_se'], c = 'darkblue', fmt ='o', label  = st.session_state.df_all[filC][' Comment'].drop_duplicates().values[0])
            # axs[1, 1].errorbar(st.session_state.df_all[filD] ['B'], st.session_state.df_all[filD]['δ11B'], yerr = st.session_state.df_all[filD]['δ11B_se'], c = 'darkorange', fmt ='o', label  = st.session_state.df_all[filD][' Comment'].drop_duplicates().values[0])
            #axs[0,0].set_ylabel('[B]_measured by Element')
            fig.text(0.5,0.04, "[B]_measurement (μg/g)", ha="center", va="center")
            fig.text(0.05,0.5, '$\u03B4^{11}$B$_{}$ (‰)', ha="center", va="center", rotation=90)
            ax.legend()
            # axs[0,1].legend()
            # axs[1,0].legend()
            # axs[1,1].legend()
            #axs.xlabel('[B]_corrected by Neptune')
            st.pyplot(fig)
            fn = 'fig.png'
            plt.savefig(fn)
            with open(fn, "rb") as img:
                btn = st.download_button(
                    label="Download image for standards",
                    data=img,
                    file_name=fn,
                    mime="image/png"
                )



    if "df_all" in st.session_state:
        st.session_state.df_all.to_csv('final.csv')
        st.write(st.session_state.df_all)
        st.write('Are you happy with these standards results? :smile:')
        st.write('You can download all results here'+':point_down:'*3)

        result_csv = st.session_state.df_all.to_csv().encode('utf-8')
        st.download_button(
            label='download results as .csv',
            data=result_csv,
            file_name='boron results.csv',
            mime='txt/csv',
        )
