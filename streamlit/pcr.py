"""
Interactive demonstration of PCR.
Author: Nathan A. Mahynski
"""
import sklearn
import scipy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.covariance import MinCovDet

from pychemauth.regressor.pcr import PCR
from pychemauth.preprocessing.scaling import RobustScaler, CorrectedScaler
from pychemauth.utils import CovarianceEllipse, OneDimLimits

from streamlit_drawable_canvas import st_canvas
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('PCR: Principal Components Regression')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of PCR for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/pca_pcr.html)
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header('Review How PCR Works')

col1_, col2_ = st.columns(2)
with col1_:
  with st.expander('Click here to see the details.'):
    st.markdown(r'''
    [Principle component regression (PCR)](https://en.wikipedia.org/wiki/Principal_component_regression) is essentially just a combination of PCA and OLS.  
    That is, we perform PCA to project X into a lower dimensional score space, then regress those scores about the target variable.  

    This can be summarized in 3 steps:

    1. Center the $X$ data for PCA,
    
    2. Project data into lower dimensional space via PCA,
    
    $$X = TP^T + E_x$$

    3. Fit the transformed data with linear regression, including an intercept term because $Y$ is not necessarily centered.  As discussed in OLS, a column of "1"s should be added as the first column in $T$ to account for the intercept term if $Y$ is not centered.

    $$Y = TQ + E_y$$
     
    $$Q = (T^TT)^{-1}T^TY$$

    The final model that makes predictions is roughly: $\hat{Y} = TQ = (XP)Q$ (but not exactly because $XP$ has a column added manually - if no intercept is used this is exact).

    PCA searches for the dimensions representing the highest degree of data variability in an unsupervised way.  However, if the response variable is not correlated with the natural "spatial" 
    variability in its regressors, $X$, then PCA will reduce the predictive ability of the model. 
''')

with col2_:
  with st.expander('Click to expand a writable canvas.'):
    col1a_, col2a_, col3a_ = st.columns(3)
    with col1a_:
      drawing_mode = st.selectbox(
          "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
      )
      if drawing_mode == 'point':
        point_display_radius = st.slider("Point display radius: ", 1, 25, 3)
    with col2a_:
      stroke_color = st.color_picker("Stroke color")
    with col3a_:
      bg_color = st.color_picker("Background color", "#eee")
    
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,
        update_streamlit=False,
        height=1320,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=True,
        key="canvas_app",
    )

st.divider() 

st.header("Upload Your Data")

uploaded_file = st.file_uploader(
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features. An example file is available [here](https://github.com/mahynski/chemometric-carpentry/blob/d4855dd5c3d1ea54048b642d6c9a4e61657a7d50/data/ols-hitters.csv).",
  type=['csv'], accept_multiple_files=False, 
  key=None, help="", 
  on_change=None, label_visibility="visible")

st.divider() 

st.header("Configure The Model")

test_size = 0.0
with st.expander("Settings"):
  
  if uploaded_file is not None:
      dataframe = pd.read_csv(uploaded_file)

      col1, col2 = st.columns(2)

      with col1:
        st.subheader("Data Settings")

        target_column = st.selectbox(label="Select a column as the target.", options=dataframe.columns, index=None, placeholder="Select a column", disabled=False, label_visibility="visible")
        feature_names = [c for c in dataframe.columns if c != target_column]
          
        random_state = st.number_input(label="Random seed for data shuffling before splitting.", min_value=None, max_value=None, value=42, step=1, placeholder="Seed", disabled=False, label_visibility="visible")
        test_size = st.slider(label="Select a positive fraction of the data to use as a test set to begin analysis.", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

      with col2:
        st.subheader("Model Settings")

        alpha = st.slider(label=r"Type I error rate ($\alpha$).", min_value=0.0, max_value=1.0, value=0.05, step=0.01, disabled=False, label_visibility="visible")
        n_components = st.slider(label="Number of dimensions to project into.", min_value=1, max_value=len(feature_names),
        value=1, step=1, disabled=False, label_visibility="visible")
        gamma = st.slider(label=r"Significance level for determining outliers ($\gamma$).", min_value=0.0, max_value=alpha, value=0.01, step=0.01, disabled=False, label_visibility="visible")
        robust = st.selectbox(label="How should we estimate $\chi^2$ degrees of freedom?", options=["Semi-Robust", "Classical"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
        robust = str(robust).lower()
        if robust == 'semi-robust':
          robust = 'semi' # Rename for PyChemAuth
        scale_x = st.toggle(label="Scale X columns by their standard deviation (it is always centered).", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        center_y = st.toggle(label="Center Y.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        scale_y = st.toggle(label="Scale Y by its standard deviation.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        sft = st.toggle(label="Use sequential focused trimming (SFT) for iterative outlier removal.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        st.write("Note: SFT relies on a Semi-Robust approach during data cleaning, then uses a Classical at the end for the final model.")

if (test_size > 0):
  X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
      dataframe[feature_names].values,
      dataframe[target_column].values,
      dataframe.index.values,
      shuffle=True,
      random_state=random_state,
      test_size=test_size,
    )

  data_tab, train_tab, test_tab, results_tab, props_tab, out_tab = st.tabs(["Original Data", "Training Data", "Testing Data", "Modeling Results", "Model Properties", "Training Set Outliers"])

  with data_tab:
    st.header("Original Data")
    st.dataframe(dataframe)

  with train_tab:
    st.header("Training Data")
    st.dataframe(pd.DataFrame(data=X_train, columns=feature_names, index=idx_train))

  with test_tab:
    st.header("Testing Data")
    st.dataframe(pd.DataFrame(data=X_test, columns=feature_names, index=idx_test))
      
  with results_tab:
    st.header("Modeling Results")

    model = PCR(
        n_components=n_components,
        alpha=alpha,
        gamma=gamma,
        scale_x=scale_x, 
        scale_y=scale_y, 
        center_y=center_y,
        robust=robust, 
        sft=sft
    )

    _ = model.fit(X_train, y_train)

    def configure_plot(ax, size=(3,3)):
      for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(6)
      fig = plt.gcf()
      fig.set_size_inches(*size)
      st.pyplot(fig, use_container_width=False)

    def fit_gaussian(data, ax):
      from scipy.stats import norm
      mu, std = norm.fit(data)
      xmin, xmax = ax.get_xlim()
      x = np.linspace(xmin, xmax, 100)
      p = norm.pdf(x, mu, std)
      ax.plot(x, p, 'k', linewidth=1, label='Guassian Fit')
      ax.axvline(mu, color='r', label=f'Gaussian Center ({"%.3f"%mu})')
      ax.legend(loc='best', fontsize=6)

    def plot_irregular(ax, model, X, y):
      extremes, outliers = model.check_xy_outliers(X, y)
      if np.sum(outliers) > 0:
        ax.plot(
            y[outliers],
            model.predict(X[outliers]),
            color='red',
            marker='x',
            ms=4,
            lw=0,
            alpha=0.5,
            label='Outliers'
        )

      if np.sum(extremes) > 0:
        ax.plot(
            y[extremes],
            model.predict(X[extremes]),
            color='yellow',
            marker='*',
            ms=4,
            lw=0,
            alpha=0.5,
            label='Extreme Values'
        )

      if np.sum(outliers) + np.sum(extremes) > 0:
        ax.legend(loc='best', fontsize=6)

      return ax

    col1sub, col2sub = st.columns([2, 2])
    with col1sub:
      st.subheader('Training Set')
        
      fig, ax = plt.subplots(nrows=1, ncols=1)
      _ = ax.plot(y_train, model.predict(X_train), 'o', ms=1)
      _ = ax.plot(y_train, y_train, '-', color='k', lw=1)
      ax.set_xlabel('Actual Value')
      ax.set_ylabel('Predicted Value')
      ax.set_title(r'Training Set ($R^2=$'+f"{'%.3f'%model.score(X_train, y_train)})")
      _ = plot_irregular(ax, model, X_train, y_train)
      configure_plot(ax)

      fig, ax = plt.subplots(nrows=1, ncols=1)
      resid = model.predict(X_train) - y_train
      _ = ax.hist(resid, bins=20, density=True)
      ax.set_xlabel(r'$y_{predicted} - y_{actual}$')
      ax.set_ylabel('Frequency')
      fit_gaussian(resid, ax)
      configure_plot(ax)

    with col2sub:
      st.subheader('Test Set')

      fig, ax = plt.subplots(nrows=1, ncols=1)
      _ = ax.plot(y_test, model.predict(X_test), 'o', ms=1)
      _ = ax.plot(y_test, y_test, '-', color='k', lw=1)
      ax.set_xlabel('Actual Value')
      ax.set_ylabel('Predicted Value')
      ax.set_title(r'Test Set ($R^2=$'+f"{'%.3f'%model.score(X_test, y_test)})")
      _ = plot_irregular(ax, model, X_test, y_test)
      configure_plot(ax)

      fig, ax = plt.subplots(nrows=1, ncols=1)
      resid = model.predict(X_test) - y_test
      _ = ax.hist(resid, bins=20, density=True)
      ax.set_xlabel(r'$y_{predicted} - y_{actual}$')
      ax.set_ylabel('Frequency')
      fit_gaussian(resid, ax)
      configure_plot(ax)

  with props_tab:
    st.write(r"$N_h = $"+f"{model._PCR__Nh_}")
    st.write(r"$N_q = $"+f"{model._PCR__Nq_}")

    st.write(r"$h_0 = $"+f"{model._PCR__h0_}")
    st.write(r"$q_0 = $"+f"{model._PCR__q0_}")
          
  with out_tab:
    st.write("If SFT is used, here are the points identified and removed from the training set.")

    if sft:
      st.dataframe(
        pd.DataFrame(data=model.sft_history['removed']['X'], columns=feature_names),
        hide_index=True
      )

      st.write('The detailed SFT history is given here:')
      st.write(model.sft_history['iterations'])

# Plot outliers in trainin / test set