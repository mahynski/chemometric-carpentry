"""
Interactive demonstration of DD-SIMCA.
Author: Nathan A. Mahynski
"""
import sklearn
from sklearn.model_selection import train_test_split
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

    target_column = st.selectbox(label="Select a column as the target class.", options=dataframe.columns, index=0, placeholder="Select a column", disabled=False, label_visibility="visible")
    target_class = st.selectbox(label="Select a class to model.", options=dataframe[target_column].unique(), index=0, placeholder="Select a class", disabled=False, label_visibility="visible")
    random_state = st.number_input(label="Random seed for data shuffling.", min_value=None, max_value=None, value=42, step=1, placeholder="Seed", disabled=False, label_visibility="visible")
    test_size = st.slider(label="Select a positive fraction of the data to use as a test set to proceed.", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

    if test_size > 0:
      X_train, X_test, y_train, y_test = train_test_split(
        dataframe[[c for c in dataframe.columns if c != target_column]],
        dataframe[target_column],
        shuffle=True,
        random_state=random_state,
        test_size=test_size,
        # stratify=dataframe[target_column]
      )

      alpha = st.slider(label="Type I Error Rate (alpha)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, disabled=False, label_visibility="visible")

      data_tab, train_tab, test_tab, results_tab = st.tabs(["Original Data", "Training Data", "Testing Data", "Modeling Results"])

      with data_tab:
        st.header("Original Data")
        st.dataframe(dataframe)

      with train_tab:
        st.header("Training Data")
        st.dataframe(X_train)

      with test_tab:
        st.header("Testing Data")
        st.dataframe(X_test)
      
      with results_tab:
        st.header("Modeling Results")
    

if __name__ == "__main__":
  print('ddsimca')
