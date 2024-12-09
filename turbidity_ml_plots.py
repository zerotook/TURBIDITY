# -*- coding: utf-8 -*-

Original file is located at
    https://colab.research.google.com/drive/...
"""

from google.colab import drive
drive.mount('/content/drive')
filepath = '/content/drive/MyDrive/Research/Colab/...'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'DejaVu Sans'

data = pd.read_csv(filepath)

X = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]   # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=61)

models = {
    'Ridge Regression': (Ridge(), {
        'alpha': [0.1, 1, 10]
    }),
    'Gradient Boosting': (GradientBoostingRegressor(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    'Random Forest': (RandomForestRegressor(random_state=61), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }),
    'Support Vector Regression': (SVR(), {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }),
    'Linear Regression': (LinearRegression(), {})
}

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.tight_layout(pad=3.0)

best_models = {}
col = 0  

for name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    _, _, _, _, r2_train = calculate_metrics(y_train, y_train_pred)
    _, _, _, _, r2_test = calculate_metrics(y_test, y_test_pred)

    ax_train = axes[0, col]
    ax_train.scatter(y_train, y_train_pred, alpha=0.7, color='#A1A9D0', edgecolor='k')
    ax_train.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax_train.set_title(f'{name}', fontsize=10)
    ax_train.set_xlabel('Augmented Observed Values', fontsize=10)
    ax_train.set_ylabel('Training Dataset \nPredicted Values', fontsize=12)
    ax_train.tick_params(axis='both', which='major', labelsize=8)

    ax_test = axes[1, col]
    ax_test.scatter(y_test, y_test_pred, alpha=0.7, color='#F0988C', edgecolor='k')
    ax_test.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax_test.set_xlabel('Augmented Observed Values', fontsize=10)
    ax_test.set_ylabel('Test Dataset \nPredicted Values', fontsize=10)
    ax_test.tick_params(axis='both', which='major', labelsize=8)

    col += 1 

plt.show()
