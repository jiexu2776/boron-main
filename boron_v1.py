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


def Process_test():
    st.session_state.tectSettingsPath = 'data/data to test/1. data folder20221129-214242'
    st.session_state.tectSettingsFolder = os.listdir(st.session_state.tectSettingsPath)





def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://github.com/jiexu2776/boron-main/blob/main/images/website-profile.gif);
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
    'https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/Goethe-Logo.gif?token=GHSAT0AAAAAACLTWTPSXR2JP5GSYBONEDBGZL3EDUA')



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


# -------------------------------
# used for mapping
# -------------------------------

def selSmpType(dataFiles):
    re_patter = r"(\d{3}\_[a-zA-Z])"


    l = []
    for file in dataFiles:
        print(file)
        match = re.search(re_patter,file)[0]
        print(match)
        l.append(match)
        #if "/" 
        #'data/data to test/1. data folder20221129-214242/file.....'
        #l.append(float(file.split("/")[-1].split('_')[0]))
    return l

# -------------------------------
# Outlier Correction and Data Conversion from str to float
# -------------------------------


def outlierCorrection_plot(data, factorSD):
    element_signal = np.array(data)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)
    fil = (data < mean + factorSD * sd) & (data > mean - factorSD * sd)
    return fil


def outlierCorrection(data, factorSD):
    element_signal = np.array(data)
    mean = np.mean(element_signal, axis=0)
    sd = np.std(element_signal, axis=0)

    return [x for x in data if (x > mean - factorSD * sd) and (x < mean + factorSD * sd)]


# -------------------------------
# background substact for 10B with the average of 9.9 and 10.2, think abut each line has to be connected
# -------------------------------

def parseBoronTable(file):
    #content = file.read()
    #print(type(file))

    if isinstance(file, str):
        with open(file, "r") as _:
            content= _.read()
        fname  = file.split('/')[-1]
        # fname = file
    else:   # streamlit file object
        content = file.getvalue().decode("utf-8")
        fname = file.__dict__["name"]
    _start = content.find("Cycle\tTime")
    _end = content.find("***\tCup")
    myTable = content[_start:_end-1]

    #cleanFname = f"temp/{fname}_cleanTable"
    
    #with open(cleanFname, "w") as _:
    #buffer = BytesIO(myTable.)

    df = pd.read_csv(StringIO(myTable),
                     sep='\t',
                     # dtype="float"   #not working -->time
                     )



    return df, fname


st.session_state.sample_plot = st.selectbox(
    'Which is your sample to plot?',
    (st.session_state.uploaded_files))


# def name_sort(data):
#     df = sorted (data)
#     for i in df:
#         res= i.plit('\')[1]



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

### function for integration data

def integrat(data):
    res = []
    res.append(data[0])
    for i in range(1, len(data)-1):
        res.append((data[i]+data[i-1]+data[i+1])/3)
    res.append(data[-1:].values[0])
    return res



def bacground_sub(factorSD, factor_B11):
    #fNames_tmp = sorted(st.session_state.fNames)
    average_B = []
    for i in st.session_state.uploaded_files:
        # if st.session_state.stage_number =  1:
        #     df_data, filename = Process_test(i)
        # else:
        df_data, filename = parseBoronTable(i)


        df_data = df_data[['Cycle', '9.9','10B', '10.2',  '10.627', '10.9' ,'11B']].astype(float)

        df_data['10.627_integrate'] = integrat(df_data['10.627'])
        df_data['10.9_integrate'] = integrat(df_data['10.9'])
        # st.write(df_data)


        df_bacground_mean = df_data[st.session_state.bac_str:st.session_state.bac_end].mean()
        df_signal = df_data[st.session_state.sig_str:st.session_state.sig_end]

    #         #substract background, substract bulc for 10B and 11B
        df_bacground_sub = df_signal - df_bacground_mean
        df_bacground_sub['10B_bulc_sub'] = df_bacground_sub['10B'] - \
            (df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2



        # df_bacground_sub['11B_bulc_sub'] = df_bacground_sub['11B'] - \
        #     (df_bacground_sub['10.627_integrate']+df_bacground_sub['10.9_integrate'])/2




        df_bacground_sub['11B_bulc_sub'] = df_bacground_sub['11B'] - \
            factor_B11*(df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
        df_bacground_sub['11B/10B'] = df_bacground_sub['11B_bulc_sub'] / \
            df_bacground_sub['10B_bulc_sub']
        fil = outlierCorrection_plot(df_bacground_sub['11B/10B'], factorSD)
        res_iso = df_bacground_sub['11B/10B'][fil]
        res_iso_outlier = df_bacground_sub['11B/10B'][~fil]
        #outliers = df_bacground_sub['11B/10B'][~res_iso]
        res_11B = outlierCorrection(df_bacground_sub['11B'], factorSD)
        if i == st.session_state.sample_plot:
            fig1, ax = plt.subplots()
            ax.plot(df_bacground_sub['11B/10B'], 'ko')
            ax.plot(res_iso_outlier, 'ro', label='outliers')
            ax.set_ylabel('$^{11}B$/$^{1O}B$')
            ax.legend()
            st.pyplot(fig1)
        #   
        average_B.append({'filename': filename, '11B': np.mean(
            res_11B), '11B/10B_row': np.mean(res_iso), 'se': np.std(res_iso)/np.sqrt(len(res_iso))})

    df = pd.DataFrame(average_B)
    st.session_state.average_B = df

    return df

    # return df

# -------------------------------
# regression based on the level from 2-5 you chosed
# -------------------------------


def polynomFit(inp, *args):
    x = inp
    res = 0
    for order in range(len(args)):
        res += args[order] * x**order
    return res


# def regression(x, y, ref_stand, order, listname):
#     #order = st.session_state.regress
#     fig2, ax = plt.subplots()
#     ax.plot(x, y, label='measuered', marker='o', linestyle='none')
#     x_use = np.array(x)
#     popt, pcov = curve_fit(polynomFit, xdata=x_use,
#                            ydata=y, 
#                            p0=[0]*(int(order)+1)
#                            )
#     fitData = polynomFit(x_use, *popt)

#     ax.plot(x_use, fitData, label='polyn. fit, order ' +
#             str(order), linestyle='--')
#     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#     ax.set_ylabel('raw data')
#     ax.set_xlabel('sequence')
#     st.pyplot(fig2)
#     res = []
#     for unknown in listname:
#         y_unknown = ref_stand / polynomFit(unknown, *popt)
#         res.append({'factor': y_unknown})
#     return(pd.DataFrame(res))





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


# -------------------------------
# data from machine for trace elements couldn't be used directly for plotting or calculation
# delete repete Ca44 and Mg26 data, unify the formate of column titles, and float all str data.
# -------------------------------


def prepare_trace(datafile):

    if 'LR' in datafile.columns[14]:
        del datafile['43Ca(LR)']
        del datafile['25Mg(LR)']
    else:
        del datafile['43Ca']
        del datafile['25Mg']

    datafile.columns = datafile.columns.str.replace('\d+', '')
    datafile.columns = datafile.columns.str.replace('\('+'LR'+'\)', '')
    res = []
    for i in range(13, len(datafile.columns)):
        for j in datafile.iloc[:, i]:
            if '<' in j:
                res.append(j)
    RES = datafile.replace(to_replace=res, value='nan', regex=True)
    RES2 = RES.replace(
        {'ERROR: Error (#1002): Internal standard composition can not be 0': np.nan})
    RES3 = RES2.replace(
        {'ERROR: Error (#1003): Calibration RM composition does not contain analyte element': np.nan})
    RES4 = RES3.iloc[:, 13:].astype(float)
    columns = RES4.iloc[:, 13:].columns
    RES4[columns] = RES4.iloc[:, 13:]
    RES4[' Sequence Number'] = RES3['LB#']
    return(RES4)


# -------------------------------
# process all isotope data, put all procedor into one function;
# -------------------------------

def processData():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.session_state.uploaded_files = st.session_state.uploaded_files

    st.header('Data Pre-Processing')

    st.subheader('Select background and signal areas')
    st.session_state.bac_str, st.session_state.bac_end = st.slider('Select background', 0, 200, (5, 70))
    st.session_state.sig_str, st.session_state.sig_end = st.slider('Select signal', 0, 200, (95, 175))
    st.pyplot(sig_selection())

    st.subheader('Set outlier and bulge factors')
    st.write('outlier factor: means data is outlier_factor times of sd will be cut')
    st.write('bulge factor: for 11B correction')
    
    col1, col2 = st.columns(2)
    with col1:
        outlier_factor = st.number_input('outlier factor', value=1.5)
    with col2:
        bulc_factor = st.number_input(
            'bulge factor', value=0.6)



    if "average_B" in st.session_state:
        #A  = st.info("Reloading already parsed dataframe!")
        df_data = st.session_state.average_B
    else:
        df_data = bacground_sub(outlier_factor, bulc_factor)


    st.subheader('Drift correction')
    st.write('Please choose your standard for boron isotopes correction')

    col1, col2 = st.columns([1, 3])

    with col1:
        standard = st.selectbox(
            'Select standard',
            ('GSD-1G', 'NIST SRM 612', 'B5', 'GSC-1G'))
        if standard == 'B5':
            number_iso = float(4.0332057)
            number_trace = float(8.42)
            SRM951_value = float(4.0492)

        if standard == 'NIST SRM 612':
            number_iso = float(4.05015)
            number_trace = float(35)
            SRM951_value = float(4.0545)

        if standard == 'GSD-1G':
            number_iso = float(4.09548)
            number_trace = float(50)
            SRM951_value = float(4.0545)

        if standard == 'GSC-1G':
            number_iso = float(4.1378)
            number_trace = float(22)
            SRM951_value = float(4.04362)

        st.session_state.standard_values = {
            "number_iso" : number_iso,
            "number_trace" : number_trace,
            "SRM951_value" : SRM951_value

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
        df_data['file name'] = selSmpType(df_data['filename'])


        s = []
        for i in df_data['file name']:
            s.append(int(i.split('_')[0]))
        df_data[' Sequence Number'] = s

        df_data.sort_values(by = [' Sequence Number'], inplace=True)#.reset_index(drop = True)

        fil = df_data['file name'].str.contains(st.session_state.sample_correction)
        df_data_B = df_data[fil]

        y_isotope = df_data_B['11B/10B_row'].astype(float)
        y_11B = df_data_B['11B'].astype(float)
        x = df_data_B[' Sequence Number']
        # x = df_data_B.index.to_numpy()
        # st.write(x)
        # get the regression function and get all corrected factors for all measurements
        #factor_iso = regression(x,y_isotope, 4.05, 4, df_data.index.to_numpy())
        #factor_B = regression(x,y_11B, 35, 4, df_data.index.to_numpy())
        factor_iso = regression(x, y_isotope,
                                number_iso,
                                st.session_state.regress_level if "regress_level" in st.session_state else st.session_state.default_reg_level,
                                df_data[' Sequence Number']
                                # df_data.index.to_numpy()
                                )

        # st.write(x)
        # get the regression function and get all corrected factors for all measurements

    # use corrected factors to correct machine drift and calculate isotope values for results
    df_data['factor_iso'] = factor_iso

    df_data['11B/10B_corrected'] = df_data['factor_iso']*df_data['11B/10B_row']
    df_data['δ11B'] = ((df_data['11B/10B_corrected']/SRM951_value)-1)*1000
    df_data['δ11B_se'] = (df_data['se']*df_data['factor_iso']/SRM951_value)*1000

##df_data_B is a dataframe for standard, df_data is a dataframe for all samples;
    st.session_state.df_data = df_data
    st.session_state.df_data_B = df_data_B


# -------------------------------
# corrected boron concerntration
# -------------------------------

def processLaser():
    if "df_data" in st.session_state:
        st.subheader('Upload your log file from Laser')
        if st.button('Test laser data'):
            st.session_state.uploaded_laser_file = 'data/data to test/2. laser file.csv'

        else:    
            st.session_state.uploaded_laser_file = st.file_uploader("Choose a laser file", type='csv')

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


# -------------------------------
# map dataframe for final results
# -------------------------------

def maping():
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
        
# -------------------------------
# main code
# -------------------------------



if len(st.session_state.uploaded_files) != 0:
    processData()

processLaser()
maping()

