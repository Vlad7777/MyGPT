#MyGPT_Least-squaresMethod


#The least-squares method is commonly used in linear regression to find the line that best fits a set of data points. In Python, you can use various libraries to perform least-squares regression. One of the most commonly used libraries is NumPy. Here's a simple example:


import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((100, 1)), X]

# Use the least squares method to find the optimal parameters (theta)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print the parameters
print("Intercept (theta0):", theta_best[0][0])
print("Slope (theta1):", theta_best[1][0])

# Predictions using the fitted model
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

# Plot the data and the regression line
plt.scatter(X, y, color='black')
plt.plot(X_new, y_predict, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

"""
In this example:

We generate synthetic data with a linear relationship.
We add a column of ones to the input features to represent the intercept term in the regression equation.
We use the least squares method to calculate the optimal parameters (theta_best) by solving the normal equation.
The parameters are then used to make predictions (y_predict) for new data points.
Finally, we visualize the data and the fitted line.
Note that for real-world applications, you would typically use machine learning libraries like scikit-learn or statsmodels for regression tasks, as they provide more functionality and are optimized for performance. However, understanding the underlying least-squares method and how to implement it using NumPy can be helpful for learning purposes.
    
"""
