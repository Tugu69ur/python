import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 7, 11, 17, 25])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

#Plot actual points and predicted curve
plt.scatter(x, y, color='blue', label='Actual Data')
x_range = np.linspace(1, 6, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred_poly = model_poly.predict(x_range_poly)
plt.plot(x_range, y_pred_poly, color='red', label='Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
#Predict x = 7
x_new = np.array([[7]])
x_new_poly = poly.transform(x_new)
y_new_pred = model_poly.predict(x_new_poly)
print("Predicted value for x=7:", y_new_pred[0])
# -------------------------------



