# Chemical Informatics Short Course

## Objectives

The goal of this short course is to introduce and explain elementary chemometric analysis methods relevant for our work at NIST.  We will also touch on more advanced ML approaches.  The course will cover the use of python-based tools that can accelerate your workflow and improve reproducibility. We will assume no prior knowledge or familiarity with any of these methods, tools, or mathematical background.  We will review only as much mathematics as is necessary to ground an understanding of the methods discussed, but a deep understanding is not necessary for application, which is the focus of this course.

What we hope to achieve:
1. Give you a new set of tools to help you do your job better
2. Create a coherent and more consistent approach to analysis within the CSD
3. Improve reproducibility and transparency

## Outline
1. Introduction
    * The Jupyter Notebook
    * The Python Language
        * Numpy, Scipy, and Matplotlib
        * watermarking
        * scitkit-learn
        * PyChemAuth
    * Common Chemometric Problems
        * N << p
        * The Authentication Problem
        <!-- OOD / class modeling vs. discriminators -->
        * Regression, Classification, and Clustering
    * Introductory Statistics
        * χ$^{2}$ statistics 
        * Baseline Performance Metrics
        <!-- 
        R^2 definition (can be < 0), vs. spearman, majority classifier
        random guessing in N dimensions -> PCA -->
2. Techniques
    * Pipelines
    * Cross-Validation
    * Class Imbalance
3. Models
    * Ordinary Least Squares (OLS)
    * Principal Components Analysis (PCA) and Regression (PCR)
    * Partial Least-Squares (PLS) or Projection to Latent Structures
    * Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis
    * Partial Least-Squares-Discriminant Analysis (PLS-DA)
    * Soft Independent Modeling of Class Analogies (SIMCA)
4. Machine Learning Techniques
    * Decision Trees
    * Ensemble Methods
    * Random Forests
    * Logistic Regression (actually classification!)
5. Inspection and Comparison
    * Comparing relative performance of pipelines
    * Model-agnostic inspection methods

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