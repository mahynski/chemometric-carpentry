"""
Interactive demonstration of DD-SIMCA.
Author: Nathan A. Mahynski
"""
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('DD-SIMCA')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python package for analysis.
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')


if __name__ == "__main__":
  print('ddsimca')
