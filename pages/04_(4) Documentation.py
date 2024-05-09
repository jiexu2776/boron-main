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




st.header('Documentation of the Boron isotope reduction program')

st.write('The code is published in Zenodo:')
link = '[![DOI](https://zenodo.org/badge/722942640.svg)](https://zenodo.org/doi/10.5281/zenodo.11150471)'
st.markdown(link, unsafe_allow_html=True)

st.write('The full documentation is realised using Quarto:')
link = '[Quarto Documentation](https://jie-xu.quarto.pub/boron-la-icp-ms-data-reduction-program/)'
st.markdown(link, unsafe_allow_html=True)

st.write('The full code is available on GitHub:')
link = '[GitHub Code Repository](https://github.com/jiexu2776/boron-main)'
st.markdown(link, unsafe_allow_html=True)

