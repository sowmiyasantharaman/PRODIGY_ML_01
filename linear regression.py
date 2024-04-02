import numpy as np
import pandas as pd

training_data = pd.read_csv('train_dataset.csv')
Xtrain = training_data[['squarefootage', 'Bathroom', 'Bedroom']].values
ytrain = training_data['SalePrice'].values


Xtrain_mean = np.mean(Xtrain, axis=0)
Xtrain_std = np.std(Xtrain, axis=0)
Xtrain = (Xtrain - Xtrain_mean) / Xtrain_std
Xtrain = np.hstack([np.ones((Xtrain.shape[0], 1)), Xtrain])
theta = np.zeros(Xtrain.shape[1])

def calculate_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient__descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        theta -= (alpha / m) * (X.T @ errors)
        cost_history.append(calculate_cost(X, y, theta))

    return theta, cost_history

alpha = 0.00005
num_iters = 1000

theta, cost_history = gradient__descent(Xtrain, ytrain, theta, alpha, num_iters)

test_data = pd.read_csv('test_dataset.csv')
Xtest = test_data[['squarefootage', 'Bathroom', 'Bedroom']].values


Xtest = (Xtest - Xtrain_mean) / Xtrain_std
Xtest = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest])

predicted_prices = Xtest @ theta

for sq_ft, price in zip(test_data['squarefootage'], predicted_prices):
    print(f'Square Footage: {sq_ft}, Predicted Price: ${price}')