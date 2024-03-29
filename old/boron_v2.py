import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as stt
from scipy import stats
from scipy.optimize import curve_fit
from io import StringIO 


st.title('Hello, welcome to the boron world')
st.sidebar.image(
    'https://raw.githubusercontent.com/Hezel2000/Data_Science/main/images/Goethe-Logo.jpg', width=150)


st.header('1. Please upload your data files from Neptune')
if st.button('clear uploaded data'):
    st.session_state.uploaded_files = []

# len(st.session_state.uploaded_files) != 0:
if 'uploaded_files' in st.session_state and len(st.session_state.uploaded_files) != 0:
    uploaded_files = st.session_state.uploaded_files

else:
    st.session_state.uploaded_files = st.file_uploader(
        'upload files', type=['.exp'], accept_multiple_files=True)


# -------------------------------
# used for mapping
# -------------------------------

def selSmpType(dataFiles):
    l = []
    for file in dataFiles:
        l.append(float(file.split('_')[0]))
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
    content = file.getvalue().decode("utf-8")
    fname = file.__dict__["name"]
    _start = content.find("Cycle\tTime")
    _end = content.find("***\tCup")
    myTable = content[_start:_end-1]

    cleanFname = f"temp/{fname}_cleanTable"
    with open(cleanFname, "w") as _:
        _.write(myTable)

    df = pd.read_csv(cleanFname,
                     sep='\t',
                     # dtype="float"   #not working -->time
                     )

    return df, fname


st.session_state.sample_plot = st.selectbox(
    'Which is your sample to plot?',
    (st.session_state.uploaded_files))


def sig_selection():
    #fNames_tmp = sorted(st.session_state.fNames)
    average_B = []
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


def bacground_sub(factorSD, factor_B11):
    #fNames_tmp = sorted(st.session_state.fNames)
    average_B = []
    for i in st.session_state.uploaded_files:
        df_data, filename = parseBoronTable(i)
        df_data = df_data[['Cycle', '9.9', '10B', '10.2', '11B']].astype(float)

        df_bacground_mean = df_data[st.session_state.bac_str:st.session_state.bac_end].mean()
        df_signal = df_data[st.session_state.sig_str:st.session_state.sig_end]

    #         #substract background, substract bulc for 10B and 11B
        df_bacground_sub = df_signal - df_bacground_mean
        df_bacground_sub['10B_bulc_sub'] = df_bacground_sub['10B'] - \
            (df_bacground_sub['9.9']+df_bacground_sub['10.2'])/2
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


def regression(x, y, ref_stand, order, listname):
    #order = st.session_state.regress
    fig2, ax = plt.subplots()
    ax.plot(x, y, label='measuered', marker='o', linestyle='none')
    x_use = np.array(x)
    popt, pcov = curve_fit(polynomFit, xdata=x_use,
                           ydata=y, 
                           p0=[0]*(int(order)+1)
                           )
    fitData = polynomFit(x_use, *popt)

    ax.plot(x_use, fitData, label='polyn. fit, order ' +
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

    st.subheader('1.1 select your background and signal area')
    st.session_state.bac_str, st.session_state.bac_end = st.slider('Select bacground', 0, 200, (5, 70))
    st.session_state.sig_str, st.session_state.sig_end = st.slider('Select signal', 0, 200, (95, 175))
    st.pyplot(sig_selection())

    st.subheader('1.2 Please set your outlier and bulge factor')
    outlier_factor = st.number_input('insert your outlier factor (means data is outlier_factor times of sd will be cut)',
                                     value=1.5)
    bulc_factor = st.number_input(
        'insert your bulge factor for 11B correction', value=0.6)



    if "average_B" in st.session_state:
        #A  = st.info("Reloading already parsed dataframe!")
        df_data = st.session_state.average_B
    else:
        df_data = bacground_sub(outlier_factor, bulc_factor)

    st.subheader(
        '1.3 Please choose your standard for boron isotopes correction')

    standard = st.selectbox(
        'NIST 612 or B5 for correction?',
        ('NIST SRM 612', 'B5'))
    if standard == 'B5':
        number_iso = int(4.0332057)
        number_trace = int(8.42)
        SRM951_value = int(4.0492)

    if standard == 'NIST SRM 612':
        number_iso = int(4.05015)
        number_trace = int(35)
        SRM951_value = int(4.0545)

    st.session_state.standard_values = {
        "number_iso" : number_iso,
        "number_trace" : number_trace,
        "SRM951_value" : SRM951_value

    }

    st.session_state.sample_correction = st.selectbox(
        'Which type is your choosed standard?',
        ('A', 'B', 'C', 'D'))

    st.session_state.default_reg_level = 4
    st.session_state.regress_level = st.number_input('insert your regression level (4 is recommended)', step=1, value=st.session_state.default_reg_level, format='%X'
                                                     )

    # Choose A/B/C/D/U to get the regression for drift correction
    fil = df_data['filename'].str.contains(st.session_state.sample_correction)
    df_data_B = df_data[fil]
    df_data[' Sequence Number'] = selSmpType(df_data['filename'])

    y_isotope = df_data_B['11B/10B_row']
    y_11B = df_data_B['11B']
    x = df_data_B.index.to_numpy()
    # st.write(x)
    # get the regression function and get all corrected factors for all measurements
    #factor_iso = regression(x,y_isotope, 4.05, 4, df_data.index.to_numpy())
    #factor_B = regression(x,y_11B, 35, 4, df_data.index.to_numpy())
    factor_iso = regression(x, y_isotope,
                            number_iso,
                            st.session_state.regress_level if "regress_level" in st.session_state else st.session_state.default_reg_level,
                            df_data.index.to_numpy()
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
        st.header('2. Please upload your log file from Laser')
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

            # sample_correction
            ref = ((st.session_state.df_map1[st.session_state.df_map1['filename'].str.contains(
                st.session_state.sample_correction)][' Spot Size (um)']/2)**2).mean()
            # #define the depth ratio
            

            st.subheader('2.1 B concerntration correction')

            #st.session_state.default_reg_level_B = 4
            st.session_state.regress_level_B = st.number_input('insert your regression level for [B] (4 is recommended)', 
            step=1, 
            value=st.session_state.default_reg_level, 
            format='%X'
                                                            )     


            y_isotope = st.session_state.df_data_B['11B/10B_row']
            y_11B = st.session_state.df_data_B['11B']
            x = st.session_state.df_data_B.index.to_numpy()   
            factor_B = regression(x, y_11B, st.session_state.standard_values["number_trace"],
                            st.session_state.regress_level_B if "regress_level_B" in st.session_state else st.session_state.default_reg_level_B, 
                            st.session_state.df_data.index.to_numpy()
                            )
            st.session_state.df_map1['factor_B'] = factor_B
            

            depth_ref = st.number_input('insert the abalation depth of selected reference / µm', value = 30.0)
            depth_sample = st.number_input('insert the abalation depth of other samples / µm', value = 30.0)

            #fil = df_map1['filename'].str.contains(sample_correction)
                    
            depth_ratios = []
            for i in st.session_state.df_map1['filename'].str.contains('A'):
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
                ref = ((st.session_state.df_map1[st.session_state.df_map1['filename'].str.contains(st.session_state.sample_correction)][' Spot Size (um)']/2)**2).mean()
                st.session_state.df_map1['[B]_corrected'] = st.session_state.df_map1['11B']*st.session_state.df_map1['factor_B'] * (ref / ((st.session_state.df_map1[' Spot Size (um)']/2)**2) / depth_ratios)

            if spot_shape == 'squre':
                dia = st.session_state.df_map1[st.session_state.df_map1['filename'].str.contains(st.session_state.sample_correction)][' Spot Size (um)']
                spotsize = dia.str.split(' ').str[0].apply(lambda x: float(x))
                st.session_state.df_map1[' Spot Size (um)'] = spotsize
                ref = ((st.session_state.df_map1[st.session_state.df_map1['filename'].str.contains(st.session_state.sample_correction)][' Spot Size (um)'])**2).mean()
                st.session_state.df_map1['[B]_corrected'] = st.session_state.df_map1['11B']*st.session_state.df_map1['factor_B'] * (ref / ((st.session_state.df_map1[' Spot Size (um)'])**2) / depth_ratios)   
            

            # #use spot diameter, depth and signal intensity to calculate [B]
            st.session_state.df_map1['[B]_corrected'] = st.session_state.df_map1['11B']*st.session_state.df_map1['factor_B'] * \
                (ref / ((st.session_state.df_map1[' Spot Size (um)']/2)**2) / depth_ratios)
    
            st.session_state.df_map1 = st.session_state.df_map1


# -------------------------------
# map dataframe for final results
# -------------------------------

def maping():
    if "df_map1" in st.session_state:
        st.subheader('2.2 export results or append your trace elements')

        trace_file = st.selectbox(
            'split stream or not?',
            ('Split stream', 'No'))

        if trace_file == 'No':
            st.session_state.df_all = st.session_state.df_map1


        elif trace_file == 'Split stream':
            st.header('3. Please upload your trace element data processed from Ladr')

            st.session_state.trace = st.file_uploader("Choose a file", type='csv', accept_multiple_files=True)
            if "trace" in  st.session_state and len(st.session_state.trace) > 0:

                trace_file = pd.read_csv(st.session_state.trace[0])

                #trace_file = pd.read_csv('2022-11-28-Si corrected-B5.csv')

                df_trace = prepare_trace(trace_file)

                st.session_state.df_all = st.session_state.df_map1.merge(df_trace, on=' Sequence Number')
                # fig4, ax = plt.subplots()
                # ax.plot([0,1],[0,1], transform=ax.transAxes, c = 'red')
                # ax.scatter(st.session_state.df_all['[B]_corrected'], st.session_state.df_all['B'], s =70, c = 'darkorange', edgecolors = 'black')
                # ax.set_ylabel('[B]_measured by Element')
                # ax.set_xlabel('[B]_corrected by Neptune')
                # st.pyplot(fig4)


        if "df_all" in st.session_state:
            st.session_state.df_all.to_csv('final.csv')
            st.write(st.session_state.df_all)
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

