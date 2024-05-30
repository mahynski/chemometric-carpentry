# A Short Course in Chemometrics

## Objectives

:dart: The goal of this short course is to introduce and explain elementary chemometric analysis methods.  We will also touch on more advanced ML approaches.  The course will cover the use of python-based tools that can accelerate your workflow and improve reproducibility. We will assume no prior knowledge or familiarity with any of these methods, tools, or mathematical background.  We will review only as much mathematics as is necessary to ground an understanding of the methods discussed since a deep understanding is not necessary for application, which is the focus of this course.

:rocket: What we hope to achieve:
1. Give you a new set of tools to help you do your job better
2. Create a coherent and more consistent approach to chemometric analysis by introducing you to a standard library for these tasks
3. Improve reproducibility and transparency

:books: In the end you will be able to go to a [library of standardized example notebooks](https://pychemauth.readthedocs.io/en/latest/applications.html), select the one you need, enter your data, then run it from start to finish.  This course will also teach you to modify and expand things as needed.

## Outline
1. Introduction
    * [The Jupyter Notebook](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/1.1_The_Jupyter_Notebook.ipynb)
        * The Basics
        * Google Colab
        * Managing Your Session
        * Installing Python Packages
        * Saving Code 
    * [The Python Language](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/1.2_The_Python_Language.ipynb)
        * Why Learn Python?
        * Before We Get Started
        * Variables
            * Built-in Data Types
            * Variable Assignment and Operators
            * Sequences: Lists, Dictionaries, and Tuples
            * Referencing
        * Logic
            * Comparison Operators
            * Logical Operators
            * If Else Statements
        * Loops
            * For Loops
            * While Loops
        * Numpy, Scipy, and Pandas
            * Numpy
            * Scipy
            * Pandas
        * Plotting with Matplotlib
        * Defining Functions
            * Documentation and Type Hints
            * Scope
            * Number and Order of Arguments
            * Default Values
        * Object Orientation and Classes
    * [Chemometrics](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/1.3_Chemometrics.ipynb)
        * The Authentication Problem
            * Some Motivating Examples
            * Class Models
            * A Machine Learning Perspective 
        * N << p
        * Regression, Classification, and Clustering
        * scitkit-learn
        * PyChemAuth
    * [Statistics Background](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/1.4_Statistics_Background.ipynb)
        * $\chi^2$ statistics 
        * Baseline Performance Metrics
        <!-- 
        R^2 definition (can be < 0), vs. spearman, majority classifier
        random guessing in N dimensions -> PCA -->
        * Rashomon sets
2. [Techniques](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/2_Techniques.ipynb)
    * Exploratory Data Analysis (EDA)
        * Basic Suggestions
        * Jensen-Shannon Divergence
            * What is it?
            * Developing an Intuition
            * JSD Reveals Plausible Tree Stumps
            * Identifying Clusters
            * Binary vs OvA
            * Common Pitfalls
    * Pipelines
    * Evaluation Metrics
    * Cross-Validation 
3. [Pre-processing](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/3_Preprocessing.ipynb)
    * Scaling and Centering
    * Filtering
       * MSC
       * SNV and RNV
       * Savitzky-Golay  
    * Missing Values and Imputation
       * Limits of Detection (LOD)
       * Basic Imputation
       * Predictive Imputers  
    * Class Balancing
       * SMOTE
       * Edited Nearest Neighbors (ENN)
       * SMOTEENN
       * ScaledSMOTEENN
       * Imblearn pipelines
    * Feature Selection 
4. [Conventional Chemometric Models](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/4_Conventional_Chemometric_Models.ipynb)
    * Regression Models
        * Ordinary Least Squares (OLS)
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/ols.html) | [sklearn API](https://scikit-learn.org/stable/modules/linear_model.html) | [Interactive Tool](https://chemometric-carpentry-ols.streamlit.app/)
        * Principal Components Analysis (PCA) and Regression (PCR)
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/pca_pcr.html) | [API](https://pychemauth.readthedocs.io/en/latest/jupyter/api/pca.html) | Interactive Tool
        * Partial Least-Squares (PLS) or Projection to Latent Structures
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/pls.html) | [API](https://pychemauth.readthedocs.io/en/latest/jupyter/api/pls.html) | Interactive Tool 
    * Classification Models
        * Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/lda.html) | [sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) | Interactive Tool 
        * Partial Least-Squares-Discriminant Analysis (PLS-DA)
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/plsda.html) | [API](https://pychemauth.readthedocs.io/en/latest/jupyter/api/plsda.html) | Interactive Tool 
        * Soft Independent Modeling of Class Analogies (SIMCA)
           * [Learn](https://pychemauth.readthedocs.io/en/latest/jupyter/learn/simca.html) | [API](https://pychemauth.readthedocs.io/en/latest/jupyter/api/simca.html) | [Interactive Tool](https://chemometric-carpentry-ddsimca.streamlit.app/)
5. [Machine Learning Models](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/5_Machine_Learning_Models.ipynb)
    * Decision Trees
    * Ensemble Methods
    * Random Forests
    * Logistic Regression (actually classification!)
    * Out-of-Distribution / Novelty Detection
6. [Inspection and Comparison](https://github.com/mahynski/chemometric-carpentry/blob/main/notebooks/6_Inspection_and_Comparison.ipynb)
    * Comparing Relative Performance of Pipelines
    * Model-agnostic Inspection Methods
    * Do I Need More Data?
7. [Case Studies](https://pychemauth.readthedocs.io/en/latest/applications.html)

<!--5. Deep Learning
    * Universal Approximation Theorem
    * Working in the Small Data Limit
        * Transfer Learning 
        * Fine Tuning
    * Embeddings
    * Convolutional Neural Nets
        * Leveraging Transfer Learning
        * Imaging Transformations
        * Out-of-Distribution Detection
    * Large Language Models
        * Transformers
            * GPT
            * BERT
        * RAG Systems
    * Chemical Foundation Models
        * Huggingface
        * ChemBERTA
    * DeepChem-->

# Next Steps:

* ❓ You can ask questions, provide feedback, and find community support on the [GitHub Discussions page](https://github.com/mahynski/chemometric-carpentry/discussions) for this course.
* ✖️ If you find a mistake please submit a [Bug Report](https://github.com/mahynski/chemometric-carpentry/issues/new/choose).
* 🔭 If you would us to cover new area(s) or have an idea to improve this course, please submit a [Feature Request](https://github.com/mahynski/chemometric-carpentry/issues/new/choose)!
* 💡 Is you have requests or ideas specific to [PyChemAuth](https://github.com/mahynski/pychemauth) you can find similar options on its [Issues page](https://github.com/mahynski/pychemauth/issues).
* 🧑‍🤝‍🧑 Please consider contributing to PyChemAuth examples!
  
---

Instructors:
* Nate Mahynski, nathan.mahynski@nist.gov
* Bill Krekelberg, william.krekelberg@nist.gov
* Tom Allison, thomas.allison@nist.gov
