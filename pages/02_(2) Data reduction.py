import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import os
import re
from io import StringIO

# --- Helper Functions ---
def add_logo():
    """Add a logo to the Streamlit sidebar."""
    st.markdown("""
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
    """, unsafe_allow_html=True)
    st.sidebar.image('https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/Goethe-Logo.gif')

def parse_boron_table(file):
    """Parse Boron isotopes table from a text file."""
    content = file.read() if isinstance(file, str) else file.getvalue().decode("utf-8")
    start = content.find("Cycle\tTime")
    end = content.find("***\tCup")
    table = content[start:end-1]
    return pd.read_csv(StringIO(table), sep='\t'), file.name if isinstance(file, StringIO) else os.path.basename(file)

def outlier_correction(data, factor_sd):
    """Filter outliers from the data based on standard deviation factor."""
    data = np.array(data)
    mean, std = np.mean(data), np.std(data)
    inlier_mask = (data > mean - factor_sd * std) & (data < mean + factor_sd * std)
    return data[inlier_mask], data[~inlier_mask]

def background_subtraction(data, bac_start, bac_end, sig_start, sig_end, factor_sd=1.5):
    """Perform background subtraction and calculate isotopic ratios."""
    data = data[['Cycle', '9.9', '10B', '10.2', '10.627', '10.9', '11B']].astype(float)
    background_mean = data[bac_start:bac_end].mean()
    signal_data = data[sig_start:sig_end] - background_mean

    # Correct isotopic data
    signal_data['10B_bulc_sub'] = signal_data['10B'] - (0.07 / 0.19) * (signal_data['9.9'] - signal_data['10.2'])
    signal_data['11B_bulc_sub'] = signal_data['11B'] - (signal_data['10.627'] + signal_data['10.9']) / 2
    signal_data['11B/10B'] = signal_data['11B_bulc_sub'] / signal_data['10B_bulc_sub']

    corrected_data, outliers = outlier_correction(signal_data['11B/10B'], factor_sd)

    return {
        'corrected_data': corrected_data,
        'mean_ratio': np.mean(corrected_data),
        'std_error': np.std(corrected_data) / np.sqrt(len(corrected_data)),
        'outliers': outliers
    }

def regression_analysis(x, y, ref_value, order, target_x):
    """Perform polynomial regression for drift correction."""
    coeffs, _ = curve_fit(lambda inp, *args: sum(arg * inp**idx for idx, arg in enumerate(args)), x, y, p0=[0] * (order + 1))
    fit_y = sum(coeff * x**idx for idx, coeff in enumerate(coeffs))
    correction_factors = ref_value / sum(coeff * target_x**idx for idx, coeff in enumerate(coeffs))
    return correction_factors, fit_y

# --- Streamlit UI ---
add_logo()

st.header('B-Isotopes Data Reduction')
st.sidebar.title("Upload Data Files")
uploaded_files = st.sidebar.file_uploader("Upload Boron data files", accept_multiple_files=True)

st.sidebar.subheader("Settings")
outlier_factor = st.sidebar.slider("Outlier Factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# Process uploaded files
if uploaded_files:
    st.subheader("Processed Data")
    results = []
    for file in uploaded_files:
        data, filename = parse_boron_table(file)
        st.write(f"Processing file: {filename}")

        # Background subtraction
        result = background_subtraction(data, bac_start=0, bac_end=10, sig_start=10, sig_end=50, factor_sd=outlier_factor)
        results.append({
            'Filename': filename,
            'Mean Ratio': result['mean_ratio'],
            'Std Error': result['std_error']
        })

        # Plot outliers
        fig, ax = plt.subplots()
        ax.plot(result['corrected_data'], 'ko', label="Inliers")
        ax.plot(result['outliers'], 'ro', label="Outliers")
        ax.axhline(y=result['mean_ratio'], color="blue", linestyle="--", label="Mean Ratio")
        ax.set_title("Outlier Detection")
        ax.legend()
        st.pyplot(fig)

    st.write("Summary Table:")
    st.dataframe(pd.DataFrame(results))
