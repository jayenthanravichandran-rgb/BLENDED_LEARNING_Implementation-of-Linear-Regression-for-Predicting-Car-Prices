# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# Algorithm
# -------------------------------
# Step 1: Start the program.
#
# Step 2: Import required libraries:
#         numpy, pandas, matplotlib, sklearn modules.
#
# Step 3: Load or create the dataset containing features such as
#         wheelbase, curbweight, enginesize, horsepower, and target price.
#
# Step 4: Display dataset preview to understand the data.
#
# Step 5: Separate independent variables (X) and dependent variable (y).
#
# Step 6: Split the dataset into training and testing sets
#         using train_test_split().
#
# Step 7: Perform feature scaling using StandardScaler
#         to normalize input features.
#
# Step 8: Apply Linear Regression:
#         a) Create LinearRegression model.
#         b) Train the model using training data.
#         c) Predict values using test data.
#
# Step 9: Evaluate Linear Regression model:
#         a) Calculate Mean Squared Error (MSE).
#         b) Calculate R² Score.
#
# Step 10: Apply Polynomial Regression:
#          a) Transform features using PolynomialFeatures (degree = 2).
#          b) Train LinearRegression model on transformed data.
#          c) Predict values using test data.
#
# Step 11: Evaluate Polynomial Regression model:
#          a) Calculate Mean Squared Error (MSE).
#          b) Calculate R² Score.
#
# Step 12: Visualize results:
#          a) Plot Actual vs Predicted prices for Linear Regression.
#          b) Plot Actual vs Predicted prices for Polynomial Regression.
#
# Step 13: Compare R² scores of both models and determine
#          which model performs better.
#
# Step 14: Display conclusion.
#
# Step 15: Stop the program. 

## Program:
```
"""
Program to implement Linear and Polynomial Regression
for predicting car prices and evaluating model performance.

Developed by: R Jayenthan
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
<img width="1002" height="501" alt="image" src="https://github.com/user-attachments/assets/665ba0de-a06d-4e8f-84df-4293626348e6" />
<img width="997" height="544" alt="image" src="https://github.com/user-attachments/assets/f64ffa62-7dcd-44bb-bef3-990639895b03" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
