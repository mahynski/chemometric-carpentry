"""
Interactive demonstration of PLS-DA.
Author: Nathan A. Mahynski
"""
import sklearn
import scipy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from streamlit_drawable_canvas import st_canvas
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('PLS-DA: PLS Discriminant Analysis')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of PLS for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/api/pls.html)

    :books: This implementation is based on [Pomerantsev, A. L., & Rodionova, O. Y. (2014). Concept and role of extreme objects in PCA/SIMCA. Journal of Chemometrics, 28(5), 429-438.](https://doi.org/10.1002/cem.2506) and also implements SFT as described in [Rodionova OY., Pomerantsev AL. "Detection of Outliers in Projection-Based Modeling",
    Analytical Chemistry 2020, 92, 2656−2664.](http://dx.doi.org/10.1021/acs.analchem.9b04611).
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header('Review How PLS-DA Works')

col1_, col2_ = st.columns(2)
with col1_:
  with st.expander('Click here to see the details.'):
    st.markdown(r'''
    [Partial least squares (PLS) regression](https://en.wikipedia.org/wiki/Partial_least_squares_regression) is also known as "projection to latent structures". There are several variations on the algorithm which are available in [sklearn](https://scikit-learn.org/stable/modules/cross_decomposition.html), however the commonly used "PLS1" (if y has one column) or "PLS2" (if y has multiple columns) are a form of [PLSRegression](https://scikit-learn.org/stable/modules/cross_decomposition.html#plsregression).

    In essence PLS is actually a scheme to project both X and Y while taking each other into account.  Assume we model **centered matrices** $X$ and $Y$ as follows:
    
    $$X = TP^T + E$$
        
    $$Y = TQ^T + F$$

    where $E$ and $F$ are error terms (assumed to be IID). $X$ has dimensions $n \times p$, and $Y$ has dimensions $n \times l$; $T$ is the $n \times k$ projection matrix of $X$, which is computed by taking both $X$ and $Y$ into account.  Here, $k \le p$ represents a dimensionality reduction; while $P$ is $p \times k$ and $Q$ is $l \times k$. 

    PLS Regression is an algorithm which can be summarized as follows (for PLS1, PLS2 deals with multiple responses instead of a single colum for $\vec{y}$):

    1. Mean-center $X$ and $Y$.  Scaling is optional.

    Then for $k$ steps:

    2. Compute the first left and right singular vectors of the covariance of matrix, $C = X^TY$, $\vec{x_w}$ and $\vec{y_w}$ (column vectors). These vectors are called `weights` - note these are single vectors corresponding to this particular iteration, $k$.  The loadings matrices, $P$ and $Q$, will be computed later.

    3. Use these weights to project $X$ to obtain the x-scores in 1D: $\vec{t} = X \vec{x_w}$.

    4. Obtain the loadings by regressing both the $X$ and $Y$ matrices using the x-scores. The $k^{\rm th}$ column in $P$ is given by $\vec{p} = X^T \vec{t} / (\vec{t}^T \vec{t})$; similarly, the $k^{\rm th}$ column in $Q$ is given by $\vec{q} = Y^T \vec{t} / (\vec{t}^T \vec{t})$.

    5. Deflate the matrices: $X \rightarrow X - \vec{t}\vec{p}^T$, $Y \rightarrow Y - \vec{t}\vec{q}^T$. 

    End loop

    6. We have now approximated $X = TP^T$ as a sum of rank-1 matrices. For convenience, we can define $XA = T$, which [provides the necessary](https://scikit-learn.org/stable/modules/cross_decomposition.html#transforming-data) transformation; sklearn refers to this as the `rotations` matrix (a similar one exists for Y). This formula is also proven [here](https://allmodelsarewrong.github.io/pls.html) in Eq. 16.20, and given on [Wikipedia](https://en.wikipedia.org/wiki/Partial_least_squares_regression):

        $$A = X_w(P^TX_w)^{-1}$$

    7. Now we can obtain a final relationship for $Y$ in terms of $X$.  Recall that since $Y$ is centered there is no intercept to worry about.

        $$Y = T Q^T = X A Q^T$$

        $$Y = X (A Q^T) = XB$$

    PLS Regression is particularly useful when (1) we have more regressors than observations ($p > n$), or (2) when the output is correlated with **dimensions that have low variance**, in which case unsupervised PCA will discard those dimensions, and with them, predictive power. 

    See the description of [DD-SIMCA](https://chemometric-carpentry-ddsimca.streamlit.app/) for an explanation of the other settings like $\gamma$ and model properties.
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

  