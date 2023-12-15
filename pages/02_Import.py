import streamlit as st


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



st.title("""Please upload your data here""")

#def Process_test():
 #   st.session_state.tectSettingsPath = 'https://github.com/jiexu2776/boron-main/tree/main/data/data to test/1. data folder20221129-214242'
  #  st.session_state.tectSettingsFolder = os.listdir(st.session_state.tectSettingsPath)
st.dataframe(pd.read_csv('./data/data to test/1. data folder20221129-214242//001_A.exp', sep='\t'))


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
