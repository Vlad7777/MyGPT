#MyGPT_SupervisedLearning

#Supervised learning is a type of machine learning where a model is trained on a labeled dataset, meaning that the input data is paired with corresponding output labels. The goal is for the model to learn the mapping between inputs and outputs so that it can make predictions on new, unseen data. Python offers various libraries for implementing supervised learning algorithms. One of the most popular ones is scikit-learn. Below is a simple example using scikit-learn for a basic supervised learning task:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predictions
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

"""
In this example:

Synthetic data is generated with a linear relationship.
The data is split into training and testing sets using train_test_split from scikit-learn.
A linear regression model is created using LinearRegression from scikit-learn.
The model is trained on the training data using the fit method.
Predictions are made on the test set using the predict method.
The model's performance is evaluated using mean squared error.
This is a basic example, and for more complex tasks, you may need to explore other supervised learning algorithms, handle categorical features, perform feature engineering, and use more advanced evaluation metrics. Scikit-learn provides a wide range of tools for various supervised learning tasks.

"""
