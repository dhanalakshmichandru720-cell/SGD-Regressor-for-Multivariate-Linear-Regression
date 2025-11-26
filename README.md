# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic multivariate data
np.random.seed(42)
n_samples = 200
X1 = np.random.rand(n_samples, 1) * 10
X2 = np.random.rand(n_samples, 1) * 5
y = 3.5 * X1 + 2.2 * X2 + 7 + np.random.randn(n_samples, 1) * 2

X = np.hstack([X1, X2])

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant')
sgd.fit(X_scaled, y.ravel())

# 4. Predictions
y_pred = sgd.predict(X_scaled)

# 5. Evaluation
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.3f}")
print(f"Coefficients: {sgd.coef_}, Intercept: {sgd.intercept_}")

# 6. Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, color='blue', label='Actual')
ax.scatter(X1, X2, y_pred, color='red', alpha=0.6, label='Predicted')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Multivariate Linear Regression with SGD')
ax.legend()
plt.show()

Developed by: dhanalakshmi.c
RegisterNumber:  25018616
*/
```

## Output:
<img width="1510" height="698" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/ae6fedeb-9b85-4e00-8fd4-8458e5103479" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
