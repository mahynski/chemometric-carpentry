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

from pychemauth.classifier.plsda import PLSDA

from streamlit_drawable_canvas import st_canvas
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('PLS-DA: PLS Discriminant Analysis')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate the use of PLS-DA for modeling data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/api/plsda.html)

    :books: This implementation is based on [Pomerantsev, A. L., & Rodionova, O. Y. (2018). Multiclass partial least squares discriminant analysis: Taking the right way - A critical tutorial. Journal of Chemometrics, 32(8), 1-16.](https://doi.org/10.1002/cem.3030).

    This tool is provided "as-is" without warranty.  See our [License](https://github.com/mahynski/chemometric-carpentry/blob/9e372e99de89ff2733d11f66c814542c37b1e2bf/LICENSE.md) for more details.
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header('Review How PLS-DA Works')

col1_, col2_ = st.columns(2)
with col1_:
  with st.expander('Click here to see the details.'):
    st.markdown(r'''
    
[The version of PLS-DA implemented here](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3030) suggests 2 approaches to PLS-DA. These are "hard" and "soft" PLS-DA, which are distinguished by how they determine their discrimination boundaries.  Both begin in the same way.  

1. One-hot encode response vector, $Y_{\rm train}$.

2. Perform PLS2 to get predictions, $\hat{Y}_{\rm train}$, after centering $X_{\rm train}$ and $Y_{\rm train}$ ($X_{\rm train}$ may also be scaled).

3. Perform PCA on $\hat{Y}_{\rm train}$ - to do this, you must first center $\hat{Y}_{\rm train}$. Project into $k-1$ dimensional space to obtain the PCA loadings matrix $T$.

4. Classify points in $T$ using either a "hard" (LDA-inspired) or "soft" (QDA-inspired) method.

Hard PLS-DA assigns a point to the class whose center is the closest in terms of Mahalanobis distance.  While it is possible to create a cutoff (based on $\chi^2$ statistics, for example) the hard version does not do this, and thus always assigns a point to one of the known classes.  Soft PLS-DA insteads uses a QDA-like distance (heteroscedastic) and uses a statistical cutoff from each class center to determine membership.  Thus, soft PLS-DA can assign a point to 0, 1, or multiple known classes.

Be wary that the definition of ["specificity" changes](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/plsda.html#Soft-PLS-DA) for soft PLS-DA from hard PLS-DA, which can make its performance appear better (numerically higher).
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
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different features. Classes should be specified as integers or strings. An example file is available [here](https://github.com/mahynski/chemometric-carpentry/blob/9e372e99de89ff2733d11f66c814542c37b1e2bf/data/plsda-2-iris.csv).",
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
        scale_x = st.toggle(label="Scale X columns by their standard deviation (it is always centered).", value=False, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")

        style = st.selectbox(label="PLS-DA style", options=["Hard", "Soft"], index=None, placeholder="Style", disabled=False, label_visibility="visible")
        if style is not None:
            style = style.lower()

if (test_size > 0) and (style is not None) and (target_column is not None):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
      dataframe[feature_names].values,
      dataframe[target_column].values,
      dataframe.index.values,
      shuffle=True,
      random_state=random_state,
      test_size=test_size,
      stratify=dataframe[target_column].values
    )

    if isinstance(y_train.dtype, (int, np.int32, np.int64)):
        not_assigned = int(dataframe[target_column].min()) - 1
    else:
        not_assigned = "UNKNOWN"
        if not_assigned in dataframe[target_column].unique():
            raise Exception("Do not use 'UNKNOWN' as class since this is used internally to denote 'no recognized class.'")

    data_tab, train_tab, test_tab, results_tab = st.tabs(["Original Data", "Training Data", "Testing Data", "Modeling Results"])

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
        model = PLSDA(
            n_components=n_components,
            alpha=alpha,
            gamma=gamma,
            not_assigned=not_assigned, 
            style=style,
            scale_x=scale_x,
            score_metric='TEFF'
        )
    
        _ = model.fit(X_train, y_train)

        def display_metrics(X, y, model):
            metrics = model.figures_of_merit(model.predict(X), y)
            col1_, col2_, col3_, col4_ = st.columns(4)
            col1_.metric(label='Total Efficiency (TEFF)', value='%.3f'%metrics['TEFF'])
            col2_.metric(label='Total Sensitivity (TSNS)', value='%.3f'%metrics['TSNS'])
            col3_.metric(label='Total Specificity (TSPS)', value='%.3f'%metrics['TSPS'])
            col4_.metric(label='Model Score', value='%.3f'%model.score(X, y))

            st.write(metrics['CM'])

        def configure_plot(ax):
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(6)
            fig = plt.gcf()
            fig.set_size_inches(2, 2)
            plt.legend(fontsize=6, bbox_to_anchor=(1,1))
            st.pyplot(fig, use_container_width=False)

        col1sub, col2sub = st.columns([2, 2])
        with col1sub:
            st.subheader('Training Set')
            display_metrics(X_train, y_train, model)

            try:
                ax = model.visualize(styles=[style], show_training=True)
                configure_plot(ax)
            except:
                pass # If > 3 classes

        with col2sub:
            st.subheader('Test Set')
            display_metrics(X_test, y_test, model)

            try:
                ax = model.visualize(styles=[style], show_training=False)
                T = model.transform(X_test)
                for i,cat in enumerate(model.categories):
                    mask = y_test == cat
                    ax.plot(T[mask], [i]*np.sum(mask), '*', color=f'C{i}', label='{} (Test)'.format(cat))
                configure_plot(ax)
            except:
                pass # If > 3 classes