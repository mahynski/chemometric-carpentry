"""
Interactive demonstration of DD-SIMCA.
Author: Nathan A. Mahynski
"""
import sklearn

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from streamlit_extras.add_vertical_space import add_vertical_space

from pychemauth.classifier.simca import SIMCA_Authenticator

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('DD-SIMCA: Data-Driven Soft Independent Modeling of Class Analogies')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of DD-SIMCA for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/api/simca.html)

    :books: This implementation is based on [Pomerantsev, A. L., & Rodionova, O. Y. (2014). Concept and role of extreme objects in PCA/SIMCA. Journal of Chemometrics, 28(5), 429-438.](https://doi.org/10.1002/cem.2506)
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

col1_, col2_ = st.columns(2)

with col1_:
  st.markdown(r'''
  How DD-SIMCA Works:

  Step 1: The raw data is **broken up by group (supervised)**; then **for each group** a PCA model for the data is constructed as follows:
      
  $$
  X = TP^T + E.
  $$

  Here, $X$, is the training data and has dimensions NxJ where $T$ is the scores matrix (projection of $X$ into some space, NxK, determined by), $P$ is the loadings matrix (JxK), and $E$ as the error or residual matrix.  $E$ may be explicitly calculated by $E = X - TP^T$. $X$ should be centered, and possibly scaled, as is required for PCA.

  Step 2: Compute the Outer Distance (squared), q, and Score Distance (squared), h, for each point where each are defined as:

  $$
  h_i = \sum_{j=1}^K \frac{t_{i,j}^2}{\lambda_j}
  $$

  $$
  q_i = \sum_{j=1}^K e_{i,j}^2.
  $$

  Step 3: Compute the critical distance for class membership. It has been [shown](https://onlinelibrary.wiley.com/doi/pdf/10.1002/cem.1147?casa_token=0NaS1t1S6mYAAAAA:VHFiiSku72EY2KXifPJtZhwXlX8PhwGPDPKUN5LvBnhB2sSTe315Uc7vlX7GmuIlgPJTNIr8chd8JA) that both the SD and OD can be well approximated by scaled chi-squared distributions.  Thus, a critical distance can be defined by a linear combination:

  $$
  c = N_h \frac{h}{h_0} + N_q \frac{q}{q_0} \sim \chi^2(N_h+N_q)
  $$

  Here, $N_h$ and $N_q$ are degrees of freedom, and $h_0$ and $q_0$ are scaling factors.  These can be estimated in a [data-driven way](https://doi.org/10.1002/cem.2506), i.e., estimated from the training set rather than fixed based on the size of the set, hence the name "DD-SIMCA."  
  The final decision rule for a class is $c < c_{crit}$ with $c_{crit} = \chi^{-2}(1-\alpha, N_h+N_q)$.
  ''')

with col2_:


st.divider() 

st.write("Start by uploading some data to model.")

uploaded_file = st.file_uploader(
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features",
  type=['csv'], accept_multiple_files=False, 
  key=None, help="", 
  on_change=None, label_visibility="visible")

st.divider() 

st.write("Next configure the model.")

with st.expander("Configure Settings"):
  
  if uploaded_file is not None:
      dataframe = pd.read_csv(uploaded_file)

      col1, col2 = st.columns(2)

      with col1:
        target_column = st.selectbox(label="Select a column as the target class.", options=dataframe.columns, index=None, placeholder="Select a column", disabled=False, label_visibility="visible")
        if target_column is not None:
          target_class = st.selectbox(label="Select a class to model.", options=dataframe[target_column].unique(), index=None, placeholder="Select a class", disabled=False, label_visibility="visible")
        random_state = st.number_input(label="Random seed for data shuffling before stratified splitting.", min_value=None, max_value=None, value=42, step=1, placeholder="Seed", disabled=False, label_visibility="visible")
        test_size = st.slider(label="Select a positive fraction of the data to use as a test set to begin analysis.", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

      with col2:
        alpha = st.slider(label="Type I error rate ($\alpha$).", min_value=0.0, max_value=1.0, value=0.05, step=0.01, disabled=False, label_visibility="visible")
        n_components = st.slider(label="Number of dimensions to project into.", min_value=1, max_value=dataframe.shape[1]-3, # account for target and index columns also
        value=1, step=1, disabled=False, label_visibility="visible")
        gamma = st.slider(label="Significance level for determining outliers ($\gamma$).", min_value=0.0, max_value=alpha, value=0.01, step=0.01, disabled=False, label_visibility="visible")
        robust = st.selectbox(label="How should we estimate $\chi^2$ degrees of freedom?", options=["Semi-Robust", "Classical"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
        robust = str(robust).lower()
        if robust == 'semi-robust':
          robust = 'semi' # Rename for PyChemAuth
        scale_x = st.toggle(label="Scale X columns by their standard deviation.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        sft = st.toggle(label="Use sequential focused trimming for iterative outlier removal.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        if target_column is not None: 
          use =  st.radio("Use a Compliant or Rigorous scoring method?", ["Rigorous", "Compliant"], captions = [f"Ignore alternatives during training (use only {target_class})", "Use alternatives to assess specificity."], index=None)
          use = str(use).lower()

if test_size > 0 and target_column is not None:
  X_train, X_test, y_train, y_test = train_test_split(
      dataframe[[c for c in dataframe.columns if c != target_column]].values,
      dataframe[target_column].values,
      shuffle=True,
      random_state=random_state,
      test_size=test_size,
      stratify=dataframe[target_column].values
  )

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

    if use is not None:
      dds = SIMCA_Authenticator(n_components=n_components, scale_x=scale_x, alpha=alpha, gamma=gamma, robust=robust, sft=sft, style='dd-simca', target_class=target_class, use=use.lower())
      _ = dds.fit(X_train, y_train)

        # _ = dds.model.extremes_plot(X_train, upper_frac=1.0)
        # fig = plt.gcf()
        # fig.set_size_inches(3,2)
        # st.pyplot(fig, use_container_width=False)

      def summary_metrics(X, y, model):
        metrics = model.metrics(X, y)
        df_t = pd.DataFrame(data=[metrics['TEFF'], metrics['TSNS'], metrics['TSPS']], columns=['Performance'], index=['Total Efficiency (TEFF)', 'Total Sensitivity (TSNS)','Total Specificity (TSPS)'])
        csps = metrics['CSPS']
        alts = list(csps.keys())
        df_c = pd.DataFrame(data=[[csps[k]] for k in alts], columns=['Class Specificity'], index=[alts])
        return df_t, df_c

      def display_metrics(X, y, model):
        metrics = model.metrics(X, y)
        col1_, col2_, col3_, col4_ = st.columns(4)
        col1_.metric(label='Total Efficiency (TEFF)', value='%.3f'%metrics['TEFF'])
        col2_.metric(label='Total Sensitivity (TSNS)', value='%.3f'%metrics['TSNS'])
        col3_.metric(label='Total Specificity (TSPS)', value='%.3f'%metrics['TSPS'])
        col4_.metric(label='Model Score', value='%.3f'%model.score(X, y))

      col1sub, col2sub = st.columns([2, 2])
      with col1sub:
        st.subheader('Training Set')
        ax = dds.model.visualize(X_train, y_train)
        plt.legend(fontsize=6, bbox_to_anchor=(1,1))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
          item.set_fontsize(6)

        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        st.pyplot(fig, use_container_width=False)

        display_metrics(X_train, y_train, dds)

      with col2sub:
        st.subheader('Test Set')
        ax = dds.model.visualize(X_test, y_test)
        plt.legend(fontsize=6, bbox_to_anchor=(1,1))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
          item.set_fontsize(6)

        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        st.pyplot(fig, use_container_width=False)

        display_metrics(X_test, y_test, dds)

        

