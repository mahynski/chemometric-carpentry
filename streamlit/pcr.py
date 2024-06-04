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
        scale_x = st.toggle(label="Scale X columns by their standard deviation.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        center_y = st.toggle(label="Center Y.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        scale_y = st.toggle(label="Scale Y by its standard deviation.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        sft = st.toggle(label="Use sequential focused trimming (SFT) for iterative outlier removal.", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
        st.write("Note: SFT relies on a Semi-Robust approach during data cleaning, then uses a Classical at the end for the final model.")

if (test_size > 0):
  X_train, X_test, idx_train, idx_test = train_test_split(
      dataframe[feature_names].values,
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

    _ = model.fit(X_train)

    def configure_plot(ax, size=(3,3)):
      for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(6)
      fig = plt.gcf()
      fig.set_size_inches(*size)
      st.pyplot(fig, use_container_width=False)

  #   ellipse_data = {}
  #   cov_ell = {}
  #   def plot_proj(ax, X, y=None, train=True, alpha=0.05, covar_method=None):
  #     fig, ax = plt.subplots(nrows=1, ncols=1)
  #     proj_ = model.transform(X)
  #     if n_components >= 2: # 2d plot
  #       if y is not None:
  #         cats = np.unique(y)
  #         for i,cat in enumerate(cats):
  #           mask = cat == y
  #           ax.plot(proj_[mask,0], proj_[mask,1], 'o', label=cat, color=f'C{i}', ms=1)
  #           if train:
  #             ellipse = CovarianceEllipse(method=covar_method).fit(proj_[mask,:2])
  #             cov_ell[i] = ellipse
  #           else:
  #             ellipse = cov_ell[i]
  #           ax = ellipse.visualize(ax, alpha=alpha, ellipse_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
  #         ax.legend(fontsize=6, loc='best')
  #       else:
  #         ax.plot(proj_[:,0], proj_[:,1], 'o')
  #         i = 0
  #         if train:
  #           ellipse = CovarianceEllipse(method=covar_method).fit(proj_[:,:2])
  #           cov_ell[i] = ellipse
  #         else:
  #           ellipse = cov_ell[i]
  #         ax = ellipse.visualize(ax, alpha=alpha, ellipse_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
  #       ax.set_xlabel(f'PC 1 ({"%.4f"%(100*model._PCA__pca_.explained_variance_ratio_[0])}%)')
  #       ax.set_ylabel(f'PC 2 ({"%.4f"%(100*model._PCA__pca_.explained_variance_ratio_[1])}%)')
  #     else:  # 1D plot
  #       if y is not None:
  #         cats = np.unique(y)
  #         for i,cat in enumerate(cats):
  #           mask = cat == y
  #           ax.plot([i+1]*np.sum(mask), proj_[mask,0], 'o', label=cat, color=f'C{i}', ms=1)
  #           if train:
  #             rectangle = OneDimLimits(method=covar_method).fit(proj_[mask,:1])
  #             cov_ell[i] = rectangle
  #           else:
  #             rectangle = cov_ell[i]
  #           ax = rectangle.visualize(ax, x=i+1-0.3, alpha=alpha, rectangle_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
  #         ax.legend(fontsize=6, loc='best')
  #       else:
  #         ax.plot([1]*np.sum(mask), proj_[:,0], 'o')
  #         i = 0
  #         if train:
  #           rectangle = OneDimLimits(method=covar_method).fit(proj_[:,0])
  #           cov_ell[i] = rectangle
  #         else:
  #           rectangle = cov_ell[i]
  #         ax = rectangle.visualize(ax, x=i+1-0.3, alpha=alpha, rectangle_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})

  #       ax.set_xlabel('Class')
  #       ax.set_xlim(0, len(cats)+1)
  #       ax.set_xticks(np.arange(1, len(cats)+1), cats, rotation=90)
  #       ax.set_ylabel(f'PC 1 ({"%.4f"%(100*model._PCA__pca_.explained_variance_ratio_[0])}%)')

  #     return ax

  #   col1sub, col2sub = st.columns([2, 2])
  #   with col1sub:
  #     ellipse_alpha = st.slider(label=r"Type I error rate ($\alpha$) for class ellipses.", min_value=0.0, max_value=1.0, value=0.05, step=0.01, disabled=False, label_visibility="visible")
  #   with col2sub:
  #     covar_method = st.selectbox("How should class covariances be computed?", ("Minimum Covariance Determinant", "Empirical"), index=0)
  #     if covar_method == "Minimum Covariance Determinant":
  #       covar_method = 'mcd'
  #     else:
  #       covar_method = 'empirical'

  #   col1sub, col2sub = st.columns([2, 2])
  #   with col1sub:
  #     st.subheader('Training Set')
  #     fig, ax = plt.subplots(nrows=1, ncols=1)
  #     ax = plot_proj(ax, X_train, None if target_column is None else y_train, train=True, alpha=ellipse_alpha, covar_method=covar_method)
  #     configure_plot(ax)

  #     fig, ax = plt.subplots(nrows=1, ncols=1)
  #     ax = model.visualize(X_train, ax=ax)
  #     ax.set_title('Training Set')
  #     ax.legend(fontsize=6, loc='upper right')
  #     configure_plot(ax)

  #   with col2sub:
  #     st.subheader('Test Set')
  #     fig, ax = plt.subplots(nrows=1, ncols=1)
  #     ax = plot_proj(ax, X_test, None if target_column is None else y_test, train=False, alpha=ellipse_alpha, covar_method=covar_method)
  #     configure_plot(ax)

  #     fig, ax = plt.subplots(nrows=1, ncols=1)
  #     ax = model.visualize(X_test, ax=ax)
  #     ax.set_title('Test Set')
  #     ax.legend(fontsize=6, loc='upper right')
  #     configure_plot(ax)

  # with props_tab:
  #   st.write(r"$N_h = $"+f"{model._PCA__Nh_}")
  #   st.write(r"$N_q = $"+f"{model._PCA__Nq_}")

  #   st.write(r"$h_0 = $"+f"{model._PCA__h0_}")
  #   st.write(r"$q_0 = $"+f"{model._PCA__q0_}")
          
  # with out_tab:
  #   st.write("If SFT is used, here are the points identified and removed from the training set.")

  #   if sft:
  #     st.dataframe(
  #       pd.DataFrame(data=model.sft_history['removed']['X'], columns=feature_names),
  #       hide_index=True
  #     )

  #     st.write('The detailed SFT history is given here:')
  #     st.write(model.sft_history['iterations'])
