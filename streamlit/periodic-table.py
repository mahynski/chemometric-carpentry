"""
Interactive demonstration of correlations between elements with an interactive periodic table.
Author: Nathan A. Mahynski
"""
import matplotlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_extras.add_vertical_space import add_vertical_space

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

from pychemauth.eda.explore import InspectData

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('Trace Element EDA')
    st.markdown('''
    ## About this application
    This tool uses the [PyChemAuth](https://pychemauth.readthedocs.io) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a teaching tool to demonstrate an aspect of EDA with trace element data.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis. For example, consider using [this notebook instead.](https://pychemauth.readthedocs.io/en/latest/jupyter/api/eda.html)
    ''')
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')

st.header("Overview")

st.markdown('''
  Collinear features can cause [confusion in models and explanation tools alike](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html). 
  It is usually best to try to decorrelate your features as much as possible before modeling.
  One way to identify and remove multicollinear features is by:
  
  1. performing [hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) on the [Spearman rank-order correlations](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) between them, 
  2. picking a threshold, and 
  3. keeping only a single feature from each cluster.

''')

st.header("Upload Your Data")

uploaded_file = st.file_uploader(
  label="Upload a CSV file. Observations should be in rows, while columns should correspond to different elements. An example file is available [here](https://github.com/mahynski/chemometric-carpentry/blob/5e7372fdad5eeec050359bff080e7557a8bdde65/data/slovenian-asparagus-trace-elements.csv).",
  type=['csv'], accept_multiple_files=False, 
  key=None, help="", 
  on_change=None, label_visibility="visible")

st.divider()

col1, col2 = st.columns(2)

if uploaded_file is not None:
  t_value = st.slider("t value", min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="t_value")

  with col1:
    # Select elements from whatever is provided.
    X = pd.read_csv(uploaded_file)
    known_elements = [str(e).lower() for e in elements.copy().symbol.values]
    used_elements = [
      str(c) for c in X.columns if str(c).lower() in known_elements
    ]
    X = X[used_elements]

    # Create the periodic table
    periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
    groups = [str(x) for x in range(1, 19)]

    df = elements.copy()
    df["atomic mass"] = df["atomic mass"].astype(str)
    df["group"] = df["group"].astype(str)
    df["period"] = [periods[x - 1] for x in df.period]
    df = df[df.group != "-"]
    df = df[df.symbol != "Lr"]
    df = df[df.symbol != "Lu"]

    # For coloring
    df["cluster"] = ["0"] * df.shape[0]

    source = ColumnDataSource(df)

    TOOLTIPS = [
      ("Name", "@name"),
      ("Atomic number", "@{atomic number}"),
      ("Atomic mass (amu) ", "@{atomic mass}"),
      ("Type", "@metal"),
      ("CPK color", "$color[hex, swatch]:CPK"),
      ("Electronic configuration", "@{electronic configuration}"),
      ("Electronegativity", "@electronegativity"),
      ("Atomic Radius (pm)", "@{atomic radius}"),
      ("Ion Radius (pm)", "@{ion radius}"),
      ("VdW Radius (pm)", "@{van der Waals radius}"),
      ("Standard State", "@{standard state}"),
      ("Bonding Type", "@{bonding type}"),
      ("Melting Point (K)", "@{melting point}"),
      ("Boiling Point (K)", "@{boiling point}"),
      ("Density (g/m^3)", "@density"),
    ]

    if 'mouseover' in st.session_state:
      hover = st.session_state.mouseover
    else:
      hover = False

    p = figure(
      title="",
      width=1000,
      height=292,
      x_range=groups,
      y_range=list(reversed(periods)),
      tools="hover" if hover else "",
      toolbar_location=None,
      tooltips=TOOLTIPS if hover else None,
    )

    # Build table as grid
    r = p.rect(
      "group",
      "period",
      0.95,
      0.95,
      source=source,
      fill_alpha=1.0,
      legend_field="cluster",
      color=factor_cmap(
        "cluster", factors=["0"], palette=["#999d9a"]
      ),
    )

    def recompute():
      """Cluster and color elements; redraw periodic table."""
      (
        selected_features,
        cluster_id_to_feature_ids,
        fig,
      ) = InspectData.cluster_collinear(
        np.asarray(X.values, dtype=np.float64),
        feature_names=X.columns,
        display=True,
        t=st.session_state.t_value, 
        highlight=False,
        figsize=(10,4)
      )

      cm_ = matplotlib.colormaps["rainbow"].resampled(
        len(cluster_id_to_feature_ids)
      )
      cmap = {"0": "#999d9a"}  # gray
      for idx, elements in sorted(
        cluster_id_to_feature_ids.items(), key=lambda x: x[0]
      ):
        cmap[str(idx)] = matplotlib.colors.rgb2hex(
          cm_(idx - 1), keep_alpha=True
        )
        for elem in elements:
          df["cluster"].where(
            ~(
              df["symbol"].apply(lambda x: str(x).lower())
              == elem.lower()
            ),
            str(idx),
            inplace=True,
          )

      df.sort_values(
        "cluster",
        inplace=True,
        key=lambda x: pd.Series([int(x_) for x_ in x]),
      )
      source.data = ColumnDataSource.from_df(df)

      r = p.rect(
          "group",
          "period",
          0.95,
          0.95,
          source=source,
          fill_alpha=1.0,
          legend_field="cluster",
          color=factor_cmap(
              "cluster",
              palette=list(cmap.values()),
              factors=list(cmap.keys()),
          ),
        )
      text_props = dict(
          source=df,  # Leave unconnected from source since this doesn't need to be updated
          text_align="left",
          text_baseline="middle",
          color="white",
      )
      x = dodge("group", -0.4, range=p.x_range)
      p.text(
          x=x,
          y="period",
          text="symbol",
          text_font_style="bold",
          **text_props
      )
      p.text(
          x=x,
          y=dodge("period", 0.3, range=p.y_range),
          text="atomic number",
          text_font_size="11px",
          **text_props
      )
      # p.text(
      #     x=x,
      #     y=dodge("period", -0.35, range=p.y_range),
      #     text="name",
      #     text_font_size="7px",
      #     **text_props
      # )
      # p.text(
      #     x=x,
      #     y=dodge("period", -0.2, range=p.y_range),
      #     text="atomic mass",
      #     text_font_size="7px",
      #     **text_props
      # )
      p.outline_line_color = None
      p.grid.grid_line_color = None
      p.axis.axis_line_color = None
      p.axis.major_tick_line_color = None
      p.axis.major_label_standoff = 0
      p.legend.orientation = "horizontal"
      p.legend.location = "top_center"
      p.hover.renderers = [r]

      st.bokeh_chart(p, use_container_width=True)

      mouseover = st.toggle('Mouseover Properties', value=False, key='mouseover')

      return fig, cluster_id_to_feature_ids

    fig, cluster_id_to_feature_ids = recompute()

  with col2:
    st.pyplot(fig, use_container_width=True)
    
  st.divider()

  st.header('Selecting Elements from Each Category')

  st.markdown('''
    Now that we have performed this clustering, how can we select which element from each cluster to actually use?
    In principle, all sets of choices should give approximately the same amount of information to a downstream model.
    However, in practice, it might be easier to actually measure certain features (here, elements) than others.  For
    example, perhaps different sets of elements are measured on different instruments and it would be advantageous to
    have a model which relies on just one instrument to provide the necessary measurements.

    Monte Carlo can be used to stochastically search for a set of features which have as many of the same "categories"
    as possible. We can manually assign categories using a funtion, like the one illustrated below.  The input should
    be the feature (element) name and the output should be a category.  For illustration, we are simply assigning 
    the category of each element as its period.  However, you can customize this function as you desire.  
    

    Refer to the [notebook here](https://pychemauth.readthedocs.io/en/latest/jupyter/api/eda.html) for more examples.
    
  ''')

  col1, col2, col3 = st.columns([3, 3, 1])

  with col2:
    n_restarts = st.slider("Number of restarts", 0, 100, value=10)
    max_iters = st.slider("Maximum iterations", 0, 100, value=10)
    T = st.slider("Temperature", 0.0, 1.0, value=0.25, step=0.05)

  with col1:
    with st.echo():
      def categorizer(symbol):
        from bokeh.sampledata.periodic_table import elements
        period = int(elements['period'][elements['symbol'] == symbol])
        return period

      best_choices = InspectData.minimize_cluster_label_entropy(
        cluster_id_to_feature_ids, # Output of InspectData.cluster_collinear
        categorizer, # Function you define above
        X=X, # Input dataframe
        seed=42, 
        n_restarts=n_restarts,
        max_iters=max_iters,
        T=T
      )

  with col3:
    st.write('Elements Selected:')
    res = pd.DataFrame(data=[[element, categorizer(element)] for element in best_choices], columns=['Element', 'Period'])
    st.dataframe(res)
