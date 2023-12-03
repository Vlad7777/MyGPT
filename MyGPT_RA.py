# My GPT
"""
Certainly! Performing regression analysis in Python is common, and one popular library for this purpose is scikit-learn. Below is a simple example of linear regression using scikit-learn. Make sure to install the library first by running:

bash
Copy code
pip install scikit-learn
Now, you can use the following Python program for a basic linear regression analysis:

"""
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lin_reg.predict(X_test)

# Plot the training data and regression line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Analysis')
plt.legend()
plt.show()

# Print the coefficients and mean squared error
print(f'Intercept: {lin_reg.intercept_}')
print(f'Coefficient: {lin_reg.coef_}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

"""
In this example, we generate some random data, split it into training and testing sets, train a linear regression model, make predictions, and visualize the results.

Please note that in a real-world scenario, you would typically use your own dataset rather than generating random data. Adjust the code accordingly based on your specific use case.

"""






