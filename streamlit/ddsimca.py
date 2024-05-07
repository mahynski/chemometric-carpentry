"""
Interactive demonstration of DD-SIMCA.
Author: Nathan A. Mahynski
"""
import pandas as pd
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('DD-SIMCA: Data-Driven Soft Independent Modeling of Class Analogies')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python package for analysis.
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.write("Start by uploading some data to model.")

uploaded_file = st.file_uploader(
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features",
  type=['csv'], accept_multiple_files=False, 
  key=None, help="", 
  on_change=None, label_visibility="visible")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(dataframe)

    target_column = st.selectbox(label="Select a column as the target class.", options=dataframe.columns, index=0, placeholder="Select a column", disabled=False, label_visibility="visible")

if __name__ == "__main__":
  print('ddsimca')
