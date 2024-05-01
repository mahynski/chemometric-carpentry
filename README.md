# A Short Course in Chemometrics

## Objectives

:dart: The goal of this short course is to introduce and explain elementary chemometric analysis methods relevant for our work at NIST.  We will also touch on more advanced ML approaches.  The course will cover the use of python-based tools that can accelerate your workflow and improve reproducibility. We will assume no prior knowledge or familiarity with any of these methods, tools, or mathematical background.  We will review only as much mathematics as is necessary to ground an understanding of the methods discussed since a deep understanding is not necessary for application, which is the focus of this course.

:rocket: What we hope to achieve:
1. Give you a new set of tools to help you do your job better
2. Create a coherent and more consistent approach to chemometric analysis across NIST by introducing you to a standard library for these tasks
3. Improve reproducibility and transparency

:books: In the end you will be able to go to a library of standardized example  notebooks, select the one you need, enter your data, then run it from start to finish.  This course will also teach you to modify and expand things as needed.

## Outline
1. Introduction
    * [The Jupyter Notebook](main/notebooks/1.1_The_Jupyter_Notebook.ipynb)
        * The Basics
        * Google Colab
        * Managing Your Session
        * Installing Python Packages
        * Saving Code 
    * [The Python Language](main/notebooks/1.2_The_Python_Language.ipynb)
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
    * [Chemometrics](main/notebooks/1.3_Chemometrics.ipynb)
        * N << p
        * The Authentication Problem
            * Some Motivating Examples
            * Class Models
            * A Machine Learning Perspective 
        * Regression, Classification, and Clustering
        * scitkit-learn
        * PyChemAuth
    * [Statistics Background](main/notebooks/1.4_Statistics_Background.ipynb)
        * $\chi^2$ statistics 
        * Baseline Performance Metrics
        <!-- 
        R^2 definition (can be < 0), vs. spearman, majority classifier
        random guessing in N dimensions -> PCA -->
        * Rashomon sets
2. [Techniques](main/notebooks/Techniques.ipynb)
    * Exploratory Data Analysis (EDA)
    * Pipelines
    * Evaluation metrics
    * Data splitting
    * Cross-Validation 
3. [Pre-processing](main/notebooks/Preprocessing.ipynb)
    * Scaling and centering
    * Imputation
    * Class balancing (SMOTE)
    * Feature selection <!-- correlation and JSD -->
3. [Conventional Chemometric Models](main/notebooks/Conventional_Chemometric_Models.ipynb)
    * Ordinary Least Squares (OLS)
    * Principal Components Analysis (PCA) and Regression (PCR)
    * Partial Least-Squares (PLS) or Projection to Latent Structures
    * Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
    * Partial Least-Squares-Discriminant Analysis (PLS-DA)
    * Soft Independent Modeling of Class Analogies (SIMCA)
4. [Machine Learning Models](main/notebooks/Machine_Learning_Models.ipynb)
    * Decision Trees
    * Ensemble Methods
    * Random Forests
    * Logistic Regression (actually classification!)
    * Out-of-distribution / novelty detection
5. [Inspection and Comparison](main/notebooks/Inspection_and_Comparison.ipynb)
    * Comparing relative performance of pipelines
    * Model-agnostic inspection methods
    * Do I need more data?

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
