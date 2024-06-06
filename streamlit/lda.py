"""
Interactive demonstration of LDA.
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pychemauth.preprocessing.scaling import RobustScaler, CorrectedScaler
from pychemauth.utils import CovarianceEllipse, OneDimLimits

from streamlit_drawable_canvas import st_canvas
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('LDA: Linear Discriminant Analysis')
    st.markdown('''
    ## About this application
    This tool uses the [scikit-learn](https://scikit-learn.org/stable/) python 
    package for analysis, but is designed to accompany [PyChemAuth](https://pychemauth.readthedocs.io) as a teaching tool.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of PCA for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/lda.html)
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header('Review How LDA Works')

col1_, col2_ = st.columns(2)
with col1_:
  with st.expander('Click here to see the details.'):
    st.markdown(r'''
    With [PCA](pca_pcr.ipynb), we found the orthogonal principal components that characterized the spread of the data, i.e., the covariance of $X$ (dimensions $n \times p$) with itself (unsupervised).  With [PLS](pls.ipynb), we looked for directions that characterized the covariance of the product of $X$ and $Y$ (supervised).  LDA is a supervised method which instead looks for axes that **maximize the separation between labelled classes**.  This is done by finding the eigenvectors ("linear discriminants") of the matrix $S_W^{-1}S_B$, where $S_W$ is the within-class scatter matrix and $S_B$ is the between-class scatter matrix.  
    
    LDA can be used as a [classification model](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/lda.html#LDA-as-a-classifier), but is more commonly used for dimensionality reduction akin to PCA. This can be described in 5 steps:

    1. Compute the $p$-dimensional mean vectors for all classes from the dataset.

    2. Compute the scatter matrices (between-class and within-class scatter matrix).

    3. Compute the eigenvectors and eigenvalues for these matrices.

    4. Sort the eigenvectors by decreasing eigenvalue and choose the first $k$ eigenvectors; stack these columns to form a $p \times k$ dimensional matrix $W$.  This is analogous to the "loadings" matrix in PCA.

    5. Use this $p \times k$ matrix to project the samples into the new subspace by performing matrix multiplication: $T = XW$, where $T$ are the "x-scores".

    PCA can be used to perform dimensionality reduction by only selecting the leading $k$ eigenvectors from the covariance matrix; assuming it is full rank, we have as many dimensions as the size of that matrix.  However, in LDA, the matrix $S_W^{-1}S_B$ will only have [at most](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Multiclass_LDA) ${\rm min}(p, c-1)$ non-zero eigenvectors where $p$ is the number of features and $c$ is the number of classes.  Thus, if we want to separate 2 classes, LDA will only be able to return the single axes that separates them best.  In such a case, it might be better to even do PCA if we desire a low, but higher than 1-, dimensional result.
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
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features. An example file is available [here](https://github.com/mahynski/chemometric-carpentry/blob/0381d097761be43e2a93465689e05ab987376a11/data/simca-iris.csv).",
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

        target_column = st.selectbox(label="Select the column which indicates class.", options=dataframe.columns, index=None, placeholder="Select a column", disabled=False, label_visibility="visible")
        feature_names = [c for c in dataframe.columns if c != target_column]

        random_state = st.number_input(label="Random seed for data shuffling before stratified splitting.", min_value=None, max_value=None, value=42, step=1, placeholder="Seed", disabled=False, label_visibility="visible")
        test_size = st.slider(label="Select a positive fraction of the data to use as a test set to begin analysis.", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

      with col2:
        st.subheader("Model Settings")

        n_components = st.slider(label="Number of dimensions to project into.", min_value=1, max_value=len(feature_names)-1,
        value=1, step=1, disabled=False, label_visibility="visible")

        standardization = st.selectbox("What type of standardization should be applied?", (None, "Scaler", "Robust Scaler"), index=0)
        if standardization is not None:
          center = st.toggle("Use centering", value=False)
          scale = st.toggle("Use scale", value=False)

if (test_size > 0) and (target_column is not None):
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

    data_tab, train_tab, test_tab, results_tab, scalings_tab = st.tabs(["Original Data", "Training Data", "Testing Data", "Modeling Results", "Scalings"])

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

        model = LDA(
            n_components=n_components,
        )

        _ = model.fit(X_train, y_train)

        def configure_plot(ax, size=(3,3)):
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(6)
            fig = plt.gcf()
            fig.set_size_inches(*size)
            st.pyplot(fig, use_container_width=False)

        cov_ell = {}
        def plot_proj(ax, X, y=None, train=True, alpha=0.05, covar_method=None):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            proj_ = model.transform(X)
            if n_components >= 2: # 2d plot
                if y is not None:
                    cats = np.unique(y)
                    for i,cat in enumerate(cats):
                        mask = cat == y
                        ax.plot(proj_[mask,0], proj_[mask,1], 'o', label=cat, color=f'C{i}', ms=1)
                        if train:
                            ellipse = CovarianceEllipse(method=covar_method).fit(proj_[mask,:2])
                            cov_ell[i] = ellipse
                        else:
                            ellipse = cov_ell[i]
                        ax = ellipse.visualize(ax, alpha=alpha, ellipse_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
                    ax.legend(fontsize=6, loc='best')
                else:
                    ax.plot(proj_[:,0], proj_[:,1], 'o')
                    i = 0
                    if train:
                        ellipse = CovarianceEllipse(method=covar_method).fit(proj_[:,:2])
                        cov_ell[i] = ellipse
                    else:
                        ellipse = cov_ell[i]
                    ax = ellipse.visualize(ax, alpha=alpha, ellipse_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
                    ax.set_xlabel(f'LD 1 ({"%.4f"%(100*model.explained_variance_ratio_[0])}%)')
                    ax.set_ylabel(f'LD 2 ({"%.4f"%(100*model.explained_variance_ratio_[1])}%)')
            else:  # 1D plot
                if y is not None:
                    cats = np.unique(y)
                    for i,cat in enumerate(cats):
                        mask = cat == y
                        ax.plot([i+1]*np.sum(mask), proj_[mask,0], 'o', label=cat, color=f'C{i}', ms=1)
                        if train:
                            rectangle = OneDimLimits(method=covar_method).fit(proj_[mask,:1])
                            cov_ell[i] = rectangle
                        else:
                            rectangle = cov_ell[i]
                            ax = rectangle.visualize(ax, x=i+1-0.3, alpha=alpha, rectangle_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})
                        ax.legend(fontsize=6, loc='best')
                else:
                    ax.plot([1]*np.sum(mask), proj_[:,0], 'o')
                    i = 0
                    if train:
                        rectangle = OneDimLimits(method=covar_method).fit(proj_[:,0])
                        cov_ell[i] = rectangle
                    else:
                        rectangle = cov_ell[i]
                    ax = rectangle.visualize(ax, x=i+1-0.3, alpha=alpha, rectangle_kwargs={'alpha':0.3, 'facecolor':f"C{i}", 'linestyle':'--'})

                    ax.set_xlabel('Class')
                    ax.set_xlim(0, len(cats)+1)
                    ax.set_xticks(np.arange(1, len(cats)+1), cats, rotation=90)
                    ax.set_ylabel(f'LD 1 ({"%.4f"%(100*model.explained_variance_ratio_[0])}%)')

            return ax

        col1sub, col2sub = st.columns([2, 2])
        with col1sub:
            ellipse_alpha = st.slider(label=r"Type I error rate ($\alpha$) for class ellipses.", min_value=0.0, max_value=1.0, value=0.05, step=0.01, disabled=False, label_visibility="visible")
        with col2sub:
            covar_method = st.selectbox("How should class covariances be computed?", ("Minimum Covariance Determinant", "Empirical"), index=0)
            if covar_method == "Minimum Covariance Determinant":
                covar_method = 'mcd'
            else:
                covar_method = 'empirical'

        col1sub, col2sub = st.columns([2, 2])
        with col1sub:
            st.subheader('Training Set')
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax = plot_proj(ax, X_train, y_train, train=True, alpha=ellipse_alpha, covar_method=covar_method)
            configure_plot(ax)

        with col2sub:
            st.subheader('Test Set')
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax = plot_proj(ax, X_test, None if target_column is None else y_test, train=False, alpha=ellipse_alpha, covar_method=covar_method)
            configure_plot(ax)

    # with scalings_tab:
    #     col1a, col2a = st.columns(2)

    #     with col1a:
    #     if n_components >= 2:
    #         fig, ax = plt.subplots(nrows=1, ncols=1)
    #         ax = model.plot_loadings(feature_names, ax=ax)
    #         for txt in ax.texts:
    #         txt.set_fontsize(6)
    #         configure_plot(ax, size=(2,2))
    #     else:
    #         fig, ax = plt.subplots()
    #         ranked_features = sorted(zip(model._PCA__pca_.components_[0], feature_names), key=lambda x:np.abs(x[0]), reverse=True)
    #         _ = ax.bar(
    #         x=np.arange(1, len(model._PCA__pca_.components_[0])+1),
    #         height=[x[0] for x in ranked_features],
    #         align='center'
    #         )
    #         ax.set_xticks(np.arange(1, len(model._PCA__pca_.components_[0])+1), [x[1] for x in ranked_features], rotation=90)
    #         configure_plot(ax, size=(int(round(len(model._PCA__pca_.components_[0])/4.)), 2))
        
    #     with col2a:
    #     fig, ax = plt.subplots()
    #     ax.plot([i+1 for i in range(len(model._PCA__pca_.components_))], model._PCA__pca_.explained_variance_ratio_.cumsum(), label='Cumulative', color='k')
    #     ax.bar(x=[i+1 for i in range(len(model._PCA__pca_.components_))], height=model._PCA__pca_.explained_variance_ratio_)
    #     ax.set_xticks([i+1 for i in range(len(model._PCA__pca_.components_))])
    #     ax.set_ylabel('Explained Variance Ratio')
    #     ax.set_xlabel('Principal Component')
    #     ax.legend(loc='best')
    #     configure_plot(ax, size=(2,2))