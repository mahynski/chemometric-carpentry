# A Short Course in Chemometrics

## Objectives

:telescope: The goal of this short course is to introduce and explain elementary chemometric analysis methods relevant for our work at NIST.  We will also touch on more advanced ML approaches.  The course will cover the use of python-based tools that can accelerate your workflow and improve reproducibility. We will assume no prior knowledge or familiarity with any of these methods, tools, or mathematical background.  We will review only as much mathematics as is necessary to ground an understanding of the methods discussed since a deep understanding is not necessary for application, which is the focus of this course.

:rocket: What we hope to achieve:
1. Give you a new set of tools to help you do your job better
2. Create a coherent and more consistent approach to chemometric analysis across NIST by introducing you to a standard library for these tasks
3. Improve reproducibility and transparency

:books: In the end you will be able to go to a library of standardized example  notebooks, select the one you need, enter your data, then run it from start to finish.  This course will also teach you to modify and expand things as needed.

## Outline
1. [Introduction](notebooks/Introduction.ipynb)
    * The [Jupyter Notebook](https://jupyter.org/)
        * The Basics
        * Google [Colab](https://colab.research.google.com/)
        * Runtime Environment
        * Installing Libraries and Tools
        * Saving Code 
    * The [Python](https://www.python.org/) Language
        * Variable Types    
        * [Numpy](https://numpy.org/), [Scipy](https://scipy.org/), and [Pandas](https://pandas.pydata.org/)
        * Plotting with [Matplotlib](https://matplotlib.org/)
        * Defining Functions
        * Object Orientation and Classes
    * Common Chemometric Problems
        * N << p
        * The Authentication Problem
        <!-- OOD / class modeling vs. discriminators -->
        * Regression, Classification, and Clustering
        * [scitkit-learn](https://scikit-learn.org/stable/index.html)
        * [PyChemAuth](https://pychemauth.readthedocs.io/en/latest/)
    * Statistics Background
        * $\chi^2$ statistics 
        * Baseline Performance Metrics
        <!-- 
        R^2 definition (can be < 0), vs. spearman, majority classifier
        random guessing in N dimensions -> PCA -->
        * Rashomon sets
2. [Techniques](notebooks/Techniques.ipynb)
    * Exploratory Data Analysis (EDA)
    * Pipelines
    * Evaluation metrics
    * Data splitting
    * Cross-Validation 
3. [Pre-processing](notebooks/Preprocessing.ipynb)
    * Scaling and centering
    * Imputation
    * Class balancing (SMOTE)
    * Feature selection <!-- correlation and JSD -->
3. [Conventional Chemometric Models](notebooks/Conventional_Chemometric_Models.ipynb)
    * Ordinary Least Squares (OLS)
    * Principal Components Analysis (PCA) and Regression (PCR)
    * Partial Least-Squares (PLS) or Projection to Latent Structures
    * Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
    * Partial Least-Squares-Discriminant Analysis (PLS-DA)
    * Soft Independent Modeling of Class Analogies (SIMCA)
4. [Machine Learning Models](notebooks/Machine_Learning_Models.ipynb)
    * Decision Trees
    * Ensemble Methods
    * Random Forests
    * Logistic Regression (actually classification!)
    * Out-of-distribution / novelty detection
5. [Inspection and Comparison](notebooks/Inspection_and_Comparison.ipynb)
    * Comparing relative performance of pipelines
    * Model-agnostic inspection methods
    * Do I need more data?

# The Future

* Feedback 
* New Topic Areas?
* Submitting bug requests to PyChemAuth
* Submitting new feature requests to PyChemAuth
* Contributing to PyChemAuth examples

---

Instructors:
* Nate Mahynski, nathan.mahynski@nist.gov
* Bill Krekelberg, william.krekelberg@nist.gov
