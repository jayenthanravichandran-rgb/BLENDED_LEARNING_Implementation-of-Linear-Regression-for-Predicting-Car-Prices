# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

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
"""
Program to implement Linear and Polynomial Regression
for predicting car prices and evaluating model performance.

Developed by: J Jayenthan
Register Number: 212225240057
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load Dataset
# -------------------------------
# You can replace this with your dataset file
# Example: df = pd.read_csv("car_data.csv")

# Sample dataset (for demonstration)
data = {
    'wheelbase': [88.6, 88.6, 94.5, 99.8, 99.4],
    'curbweight': [2548, 2548, 2823, 2337, 2824],
    'enginesize': [130, 130, 152, 109, 136],
    'horsepower': [111, 111, 154, 102, 115],
    'price': [13495, 16500, 16500, 13950, 17450]
}

df = pd.DataFrame(data)

print("\nDataset Preview:")
print(df.head())

# -------------------------------
# 2. Define Features & Target
# -------------------------------
X = df.drop('price', axis=1)
y = df['price']

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 5. Linear Regression
# -------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

# Evaluation
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("\nLinear Regression Results:")
print("MSE:", mse_lin)
print("R2 Score:", r2_lin)

# -------------------------------
# 6. Polynomial Regression
# -------------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression Results:")
print("MSE:", mse_poly)
print("R2 Score:", r2_poly)

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure()
plt.scatter(y_test, y_pred_lin)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

plt.figure()
plt.scatter(y_test, y_pred_poly)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Polynomial Regression: Actual vs Predicted")
plt.show()

# -------------------------------
# 8. Conclusion
# -------------------------------
if r2_poly > r2_lin:
    print("\nPolynomial Regression performs better.")
else:
    print("\nLinear Regression performs better.")
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
