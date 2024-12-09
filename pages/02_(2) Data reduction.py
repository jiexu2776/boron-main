import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import re
from io import StringIO

# --- UI Enhancements ---
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

st.sidebar.image(
    'https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/Goethe-Logo.gif'
)
st.header('B-Isotopes Data Reduction')

# --- Utility Functions ---
def parse_boron_table(file):
    """
    Extracts boron data table from a file.
    """
    content = file.getvalue().decode("utf-8") if not isinstance(file, str) else open(file, "r").read()
    fname = file.name if hasattr(file, "name") else file.split('/')[-1]
    start_idx = content.find("Cycle\tTime")
    end_idx = content.find("***\tCup")
    table_data = content[start_idx:end_idx-1]
    return pd.read_csv(StringIO(table_data), sep='\t'), fname


def remove_outliers(data, factor_sd):
    """
    Removes outliers based on a given standard deviation factor.
    """
    mean = np.mean(data)
    sd = np.std(data)
    mask = (data > mean - factor_sd * sd) & (data < mean + factor_sd * sd)
    return data[mask], data[~mask]


def extract_sample_type(filenames):
    """
    Extracts sample types based on a regex pattern.
    """
    return [re.search(r"(\d{3}\_[a-zA-Z])", fname)[0] for fname in filenames]


def subtract_background(data, start, end):
    """
    Performs background subtraction for boron data.
    """
    background_mean = data[start:end].mean()
    signal = data.copy() - background_mean
    signal['10B_bulc_sub'] = signal['10B'] - (0.07/0.19)*(signal['9.9'] - signal['10.2'])
    signal['11B_bulc_sub'] = signal['11B'] - (signal['10.627'] + signal['10.9']) / 2
    signal['11B/10B'] = signal['11B_bulc_sub'] / signal['10B_bulc_sub']
    return signal


def polynomial_fit(x, *coeffs):
    """
    Polynomial function for curve fitting.
    """
    return sum(c * x**i for i, c in enumerate(coeffs))


def perform_regression(x, y, ref_value, order):
    """
    Fits a polynomial regression and calculates correction factors.
    """
    coefficients, _ = curve_fit(polynomial_fit, x, y, p0=[0] * (order + 1))
    fit_values = polynomial_fit(x, *coefficients)
    return ref_value / fit_values


# --- Main Data Processing ---
st.subheader('Set Outlier Threshold')
outlier_factor = st.number_input('Outlier Factor (SD)', value=1.5)

if "processed_data" not in st.session_state:
    uploaded_files = st.file_uploader("Upload Boron Data Files", accept_multiple_files=True)
    if uploaded_files:
        processed_data = []
        for file in uploaded_files:
            raw_data, filename = parse_boron_table(file)
            raw_data = raw_data.astype(float)
            raw_data = raw_data[['Cycle', '9.9', '10B', '10.2', '10.627', '10.9', '11B']]

            signal_data = subtract_background(raw_data, 10, 20)
            filtered_data, outliers = remove_outliers(signal_data['11B/10B'], outlier_factor)
            mean_11B = filtered_data.mean()
            processed_data.append({'filename': filename, 'mean_11B': mean_11B})
        
        st.session_state.processed_data = pd.DataFrame(processed_data)

# Display Results
if "processed_data" in st.session_state:
    st.write("Processed Data:")
    st.dataframe(st.session_state.processed_data)
