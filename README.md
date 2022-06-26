# Machine Learning Regression with Kaggle House Prices

This is a deep dive into learning to solve machine learning regression problems.  Supervised learning with a continuous target value.

The data set used is from the the Kaggle Competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Given several dozen predictors/featues, we want to accurately predict the sale price of a house.

## Notebooks

- [Quickstart](house-prices-quickstart.ipynb)
- [Lasso and Ridge Regression](house-prices-lasso-and-ridge.ipynb)
- [Polynomial Features](house-prices-polynomial.ipynb)
- [Stochastic Gradient Descent](house-prices-sgd.ipynb)
- [Decision Tree and Random Forests](house-prices-decision-tree-and-random-forest.ipynb)
- ***
- [XGBoost](house-prices-xgboost.ipynb)
- [Tensorflow](house-prices-tensorflow.ipynb)
- [PyTorch](house-prices-pytorch.ipynb)
- [MLPRegressor]
- [EDA](house-prices-eda.ipynb)
- Feature Engineering

## Machine Learning Models Covered

- Linear Regression
  - Lasso - L1
  - Ridge - L2
  - Polynomial
  - Residuals
  - Collinearity
  - Interactions
  - Mathematics
    - Solving Ax=b using numpy
    - Normal Equations
- Decision Trees  
- Gradient Boosted Decision Trees (GBDT)
- Support Vector Machines
- [Principal Component Analysis](pca.md) (PCA)
- Stochastic Gradient Descent
- Deep Neural Networks (DNN)
  - Activation Functions

In addition, we will cover other topics important to solving machine:

- Feature Engineering
  - Data Transformation
    - Scaling
    - Gaussian Normal
    - log transform
    - skew, kurtosis
- Missing Values
- Loss Functions
  - MAE
  - RMSE
  - Huber
- Feature Selection
  - Forward Selection
  - Reverse Selection
  - SHAP
  - Permutation Importance
  - Mutual Information
- Hyperparameter Optimization

## RMSE

[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

$$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{\hat{y}_i -y_i}{\sigma_i}\Big)^2}}$$

