# Finding Donors for CharityML - Supervised Learning

Finding-donors is a supervise learning project 

## Getting Started
In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.

## Data Source
The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi online. The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.

## Things you will learn by completing this project:

- How to explore data and observe features.
- How to train and test models.
- How to identify potential problems, such as errors due to bias or variance.
- How to apply techniques to improve the model, such as cross-validation and grid search.

##  Things you will learn by completing this project:
- Exploring data
- Transforming data
- Normalizing numerical features
- Shuffle and Split data
- Evaluate Model Performance
- Performance Metrics
- Comparison of Model Performances
- Model Tuning
- Feature Importance ranking
- You may explore **Supervised Learning Models** 
  - **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
    - Gaussian Naive Bayes (GaussianNB)
    - Decision Trees
    - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
    - K-Nearest Neighbors (KNeighbors)
    - Stochastic Gradient Descent Classifier (SGDC)
    - Support Vector Machines (SVM)
    - Logistic Regression

## Software and Libraries
This project uses the following software and Python libraries:

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) 
distribution of Python, which already has the above packages and more included. 
Make sure that you can select the Python 2.x installer and or the Python 3.x installer. This particular notebook uses Python 3.6.

## About the Files

This project contains three files:

- `finding_donors.ipynb`: This is the main file where you will be performing your work on the project.
- `census.csv`: The project dataset. You'll load this data in the notebook.
- `visuals.py`: This Python script provides supplementary visualizations for the project. Do not modify.

## Reference:

You can find the `finding_donors` folder containing the necessary project files on the [Machine Learning projects GitHub](https://github.com/udacity/machine-learning), under the `projects` folder. 
You may download all of the files for projects we'll use in this Nanodegree program directly from this repo.
