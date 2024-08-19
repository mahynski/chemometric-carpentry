"""
Interactive demonstration of OLS Regression.
Author: Nathan A. Mahynski
"""
import sklearn

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from pychemauth.preprocessing.scaling import RobustScaler, CorrectedScaler

from streamlit_drawable_canvas import st_canvas
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('OLS: Ordinary Least Squares Regression')
    st.markdown('''
    ## About this application
    This tool uses the [scikit-learn](https://scikit-learn.org/stable/) python 
    package for analysis, but is designed to accompany [PyChemAuth](https://pychemauth.readthedocs.io) as a teaching tool.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of OLS for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/ols.html)

    This tool is provided "as-is" without warranty.  See our [License](https://github.com/mahynski/chemometric-carpentry/blob/9e372e99de89ff2733d11f66c814542c37b1e2bf/LICENSE.md) for more details.
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header('Review How OLS Works')

col1_, col2_ = st.columns(2)
with col1_:
  with st.expander('Click here to see the details.'):
    st.markdown(r'''
      OLS stands for [ordinary least squares regression](https://en.wikipedia.org/wiki/Ordinary_least_squares). This is generally considered to be the simplest of all regression methods.

      We want to develop a model for some scalar response variable, $\vec{y}$, in terms of $p$ explanatory variables, or features stored in a matrix $X$.  The dimensions of $X$ are $n \times p$; assume $\vec{y}$ is a column vector ($n \times 1$) with a single scalar value for each row in $X$.

      $$y_0 = b_0 +b_1x_{0,1}+\dots b_px_{0,p}$$

      $$y_1 = b_0 +b_1x_{1,1}+\dots b_px_{1,p}$$

      $$\dots$$

      In matrix form we can write $\vec{y} = X\vec{b} + \vec{e}$, where $\vec{e}$ is some error and $\vec{b}$ is a vector of coefficients of the size ($p \times 1$). $\vec{b}$ needs to be solved for.  This is done by minimizing the square error, $err = \sum (\vec{y} - X\vec{b})^2$, hence the name "least squares regression".  To solve this we can take the derivative and set it equal to zero.  Thus, we arrive at:

      $$-2X^T(\vec{y} - X\vec{b}) = 0,$$

      $$X^T\vec{y} = X^TX\vec{b},$$

      $$\vec{b} = (X^TX)^{-1}X^T\vec{y}.$$

      This is equivalent to finding the $\vec{b}$ that minimizes the loss, $L$:

      $$
      L = ||X\vec{b} -\vec{y} ||_2^2.
      $$

      However, when we have many terms (e.g., $p > n$) we can start to overfit to our model. Overfitting is often characterized by coefficients with large values, for example, consider polynomials functions with large coefficients of alternating sign.  This insight is what has given rise to several forms of regularization:

      * L2 or ["ridge" regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
      * L1 or ["LASSO" regression](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

      In both cases, instead of simply minimizing the error between the model and the observations (pure OLS) another term is added to $L$ to prevent the model from overfitting.  We will explore both below.
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
        height=650,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=True,
        key="canvas_app",
    )

st.divider() 

st.header("Upload Your Data")

uploaded_file = st.file_uploader(
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features. An example file is available [here](https://github.com/mahynski/chemometric-carpentry/blob/7401f28144ebf60a7d4a9455d166eddabbf5e78f/data/ols-hitters.csv).",
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

        standardization = st.selectbox("What type of standardization should be applied?", (None, "Scaler", "RobustScaler"), index=0)
        if standardization is not None:
          center = st.toggle("Use centering", value=False)
          scale = st.toggle("Use scale", value=False)

      with col2:
        st.subheader("Model Settings")

        # select regularization type
        reg_type = st.selectbox("What type of regularization should be applied?", (None, "LASSO (L1)", "Ridge (L2)"), index=0)

        # select strength
        if reg_type is not None:
          reg_strength = st.select_slider("Regularization strength", options=np.logspace(-6, 6, 25))


if (test_size > 0):
  X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    dataframe[feature_names].values,
    dataframe[target_column].values,
    dataframe.index.values,
    shuffle=True,
    random_state=random_state,
    test_size=test_size,
  )

  if standardization == "Scaler":
    scaler = CorrectedScaler(with_mean=center, with_std=scale)
  elif standardization == "RobustScaler":
    scaler = RobustScaler(with_median=center, with_iqr=scale)
  else:
    scaler = CorrectedScaler(with_mean=False, with_std=False)

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  data_tab, train_tab, test_tab, results_tab, coef_tab = st.tabs(["Original Data", "Training Data", "Testing Data", "Modeling Results", "Coefficients"])

  with data_tab:
    st.header("Original Data")
    st.dataframe(dataframe)

  with train_tab:
    st.header("Training Data")
    st.dataframe(pd.DataFrame(data=np.hstack((X_train, y_train.reshape(-1,1))), columns=feature_names+[target_column], index=idx_train))

  with test_tab:
    st.header("Testing Data")
    st.dataframe(pd.DataFrame(data=np.hstack((X_test, y_test.reshape(-1,1))), columns=feature_names+[target_column], index=idx_test))
      
  with results_tab:
    st.header("Modeling Results")

    if reg_type is None:
      model = linear_model.LinearRegression(fit_intercept=True)
    elif reg_type == "LASSO (L1)":
      model = linear_model.Lasso(alpha=reg_strength, fit_intercept=True)
    else:
      model = linear_model.Ridge(alpha=reg_strength, fit_intercept=True)

    _ = model.fit(X_train, y_train)

    def configure_plot(ax, size=(2,2)):
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

    col1sub, col2sub = st.columns([2, 2])
    with col1sub:
      st.subheader('Training Set')
        
      fig, ax = plt.subplots(nrows=1, ncols=1)
      _ = ax.plot(y_train, model.predict(X_train), 'o', ms=1)
      _ = ax.plot(y_train, y_train, '-', color='k', lw=1)
      ax.set_xlabel('Actual Value')
      ax.set_ylabel('Predicted Value')
      ax.set_title(r'Training Set ($R^2=$'+f"{'%.3f'%model.score(X_train, y_train)})")
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
      configure_plot(ax)

      fig, ax = plt.subplots(nrows=1, ncols=1)
      resid = model.predict(X_test) - y_test
      _ = ax.hist(resid, bins=20, density=True)
      ax.set_xlabel(r'$y_{predicted} - y_{actual}$')
      ax.set_ylabel('Frequency')
      fit_gaussian(resid, ax)
      configure_plot(ax)

  with coef_tab:
    fig, ax = plt.subplots()
    ranked_features = sorted(zip(model.coef_, feature_names), key=lambda x:np.abs(x[0]), reverse=True)
    _ = ax.bar(
      x=np.arange(1, len(model.coef_)+1),
      height=[x[0] for x in ranked_features],
      align='center'
    )
    ax.set_xticks(np.arange(1, len(model.coef_)+1), [x[1] for x in ranked_features], rotation=90)
    configure_plot(ax, size=(int(round(len(model.coef_)/4.)),2))
