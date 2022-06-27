# Machine Learning Regression with Kaggle House Prices

This is a deep dive into learning to solve machine learning regression problems.  Supervised learning with a continuous target value.

The data set used is from the the Kaggle Competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Given several dozen predictors/featues, we want to accurately predict the sale price of a house.

## Notebooks

- [Quickstart](house-prices-quickstart.ipynb)
- [Lasso and Ridge Regression](house-prices-lasso-and-ridge.ipynb)
- [Polynomial Features](house-prices-polynomial.ipynb)
- [Label Encoding and Imputation]
- Residuals pd.DataFrame({'Error Values': (y_test - pred)}).hvplot.kde()
- [SelectK Best Features]
- [Forward Feature Selection]
- [Stochastic Gradient Descent](house-prices-sgd.ipynb)
- [Outliers]
- [Decision Tree and Random Forests](house-prices-decision-tree-and-random-forest.ipynb)
- [GridSearchCV]
- [Gradient Boosted Trees - XGBoost/Catboost/LightGBM](house-prices-xgboost.ipynb)
- [GBDT Feature Importance]
- [SHAP Values]
- [Data Transformation]
- [Support Vector Machines](house-prices-support-vector-regression.ipynb)
- [Tensorflow](house-prices-tensorflow.ipynb)
- ***
- [Nonlinear Regression]
- [Ensemble Learning - Blending]
- [Robust Regression - RANSAC]
- [PyTorch](house-prices-pytorch.ipynb)
- [MLPRegressor]
- [Basic EDA](house-prices-eda.ipynb)
- [Enhanced EDA]
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

In addition, we will cover other topics important to machine learning:

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

