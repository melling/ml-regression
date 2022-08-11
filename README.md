This is a collection of repos devoted to learning machine learning with Kaggle.  

- [Regression with Housing Prices](https://github.com/melling/ml-regression)
- [Classification with Titanic](https://github.com/melling/ml-kaggle-titanic)
- [MNIST Solutions](https://github.com/melling/ml-mnist-kaggle-digit-recognizer)
- [Classification with Spaceship Titanic](https://github.com/melling/ml-kaggle-spaceship-titanic)
- [NLP with Disaster Tweets](https://github.com/melling/ml-nlp-kaggle-disaster-tweets)
- 

Follow me on [Kaggle](https://www.kaggle.com/mmellinger66/)

# Machine Learning Regression with Kaggle House Prices

This is a deep dive into learning to solve machine learning regression problems.  Supervised learning with a continuous target value.

The data set used is from the the Kaggle Competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Given several dozen predictors/featues, we want to accurately predict the sale price of a house.

## Notebooks

- [Quickstart](house-prices-quickstart.ipynb)
- [Lasso, Ridge, and ElasticNet Regression](house-prices-lasso-and-ridge.ipynb)
- [Polynomial Features](house-prices-polynomial.ipynb)
- [Target and Feature Distributions](house-prices-target-feature-distributions.ipynb)
- [Simple Imputer and Label Encoding](house-prices-simple-imputer.ipynb)
- [Robust Regression - RANSAC](house-prices-robust-regression.ipynb)
- [SelectK Best Features]
- Variance Inflation Factor (VIF)
- Recursive Feature Elimination (RFE)
- Mutual Information Gain
- [Forward Feature Selection]
- [Stochastic Gradient Descent](house-prices-sgd.ipynb)
- [Lasso, Ridge, and ElasticNet with log(target)](house-prices-lasso-ridge-log-target.ipynb)
- [Outliers]
- [Decision Tree and Random Forests](house-prices-decision-tree-and-random-forest.ipynb)
- [GridSearchCV](house-prices-rf-gridsearchcv.ipynb)
- [MLPRegressor](house-prices-mlpregressor.ipynb)
- [Gradient Boosted Trees - XGBoost/Catboost/LightGBM](house-prices-xgboost.ipynb)
- [GBDT Feature Importance]
- [SHAP Values]
- [XGBoost + CV with OOF Results]
- [XGBoost + Optuna]
- [Data Transformation]
- [Support Vector Machines](house-prices-support-vector-regression.ipynb)
- [Tensorflow](house-prices-tensorflow.ipynb)
- [KerasTuner]
- ***
- [Target Encoding]
- [Ensemble Learning - Blending]
- [Ensemble Learning - Stacking]
- [Robust Regression - RANSAC]
- [Nonlinear Regression]
- [PyTorch](house-prices-pytorch.ipynb)
- [Basic EDA](house-prices-eda.ipynb)
- [Enhanced EDA]
- Feature Engineering
- [Linear Regression from Scratch](house-prices-lr-from-scratch.ipynb)

## Misc Notebooks

- [DSML Feature Selection]


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
- Outliers
  - Z-score
  - IQR Method
  - https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer
  - Hypothesis Testing
  - DBSCAN Clustering
- Loss Functions
  - MAE
  - RMSE
  - Huber
- Feature Selection
  - Forward Selection
  - Reverse Selection
  - SHAP
    - https://h2o.ai/blog/shapley-values-a-gentle-introduction/
  - Permutation Importance
  - Mutual Information
- Hyperparameter Optimization

## MAE


[Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

$$MAE = \frac{\sum_{i=1}^n |y_i - x_i|}{n}$$

## RMSE

[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

$$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{\hat{y}_i -y_i}{\sigma_i}\Big)^2}}$$

