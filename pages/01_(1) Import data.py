import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO

# Constants
LOGO_URL = "https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/Goethe-Logo.gif"
PROFILE_GIF_URL = "https://raw.githubusercontent.com/jiexu2776/boron-main/main/images/website-profile.gif"
TEST_DATA_PATH = "data/data to test/1. data folder20221129-214242"
LASER_FILE_PATH = "data/data to test/2. laser file.csv"

# App Configuration
st.set_page_config(page_title="Boron Data Analysis", layout="wide")

# Sidebar Logo
st.sidebar.image(LOGO_URL, use_column_width=True)

# Custom Styles
def apply_custom_styles():
    """Apply custom styles for buttons and sidebar."""
    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage] {
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
            .stButton > button {
                color: black;
                background: lightblue;
                width: 200px;
                height: 50px;
            }
            [data-testid="stSidebarNav"] {
                background-image: url(""" + PROFILE_GIF_URL + """);
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
        unsafe_allow_html=True
    )

apply_custom_styles()

# Helper Functions
def parse_boron_table(file):
    """Parse a boron table from the provided file."""
    if isinstance(file, str):
        with open(file, "r") as f:
            content = f.read()
        fname = os.path.basename(file)
    else:
        content = file.getvalue().decode("utf-8")
        fname = file.name

    start_idx = content.find("Cycle\tTime")
    end_idx = content.find("***\tCup")
    table_content = content[start_idx:end_idx - 1]

    try:
        df = pd.read_csv(StringIO(table_content), sep='\t')
        return df, fname
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None, None

def process_test_data():
    """Load test data for processing."""
    st.session_state.tectSettingsFolder = os.listdir(TEST_DATA_PATH)

def plot_signal_selection():
    """Generate a signal selection plot."""
    df_data, _ = parse_boron_table(st.session_state.sample_plot)
    df_data = df_data[['Cycle', '9.9', '10B', '10.2', '11B']].astype(float)

    fig, ax = plt.subplots()
    ax.plot(df_data['11B'], label='11B', color='green')
    ax.plot(df_data['10B'], label='10B', color='firebrick')
    ax.set_ylabel('Signal Intensity')
    ax.set_xlabel('Cycle')

    x = df_data['11B'].index.to_numpy()
    ax.fill_between(
        x, max(df_data['11B']),
        where=(x < st.session_state.sig_end) & (x > st.session_state.sig_str),
        alpha=0.5
    )
    ax.fill_between(
        x, max(df_data['11B']),
        where=(x < st.session_state.bac_end) & (x > st.session_state.bac_str),
        alpha=0.5
    )
    ax.legend()
    return fig

# Main Application
st.title("Boron Data Analysis Tool")
st.subheader("1. Upload Neptune Experiment Files")

if st.button("Load Test Data"):
    process_test_data()
    st.session_state.uploaded_files = [
        os.path.join(TEST_DATA_PATH, file)
        for file in st.session_state.tectSettingsFolder
        if file.endswith(".exp")
    ]
    st.write("You have uploaded test dataset")

if st.button("Clear Uploaded Data"):
    st.session_state.uploaded_files = []

uploaded_files = st.session_state.get("uploaded_files", st.file_uploader(
    "Upload Experiment Files", type=["exp"], accept_multiple_files=True
))

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    st.session_state.sample_plot = st.selectbox("Select a Sample to Plot:", uploaded_files)

    # Background and Signal Selection
    st.session_state.bac_str, st.session_state.bac_end = st.slider(
        "Select Background Range", 0, 200, (5, 70)
    )
    st.session_state.sig_str, st.session_state.sig_end = st.slider(
        "Select Signal Range", 0, 200, (95, 175)
    )

    st.pyplot(plot_signal_selection())

st.subheader("2. Upload Laser Log File")
if st.button("Test Laser Data"):
    st.session_state.uploaded_laser_file = LASER_FILE_PATH
    st.write("You have uploaded test dataset")
else:
    st.session_state.uploaded_laser_file = st.file_uploader(
        "Upload Laser File", type="csv")
    st.write("You have uploaded you own data")
