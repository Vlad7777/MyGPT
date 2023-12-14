#MyGPT_RA_Iris

"""
Performing regression analysis using the Iris dataset involves predicting a continuous variable based on one or more independent variables. However, the Iris dataset is typically used for classification tasks because it consists of features (sepal length, sepal width, petal length, petal width) that classify iris flowers into different species.

If you have a specific continuous target variable in mind that you want to predict, you can still use regression techniques. For this example, let's assume you want to predict the petal length based on other features.

Here's a basic example using linear regression in Python with the Iris dataset:

"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

#iris_df
# Printing top 5 rows
iris_df.head()




# Let's use petal width and sepal length as independent variables for simplicity
X = iris_df[['sepal length (cm)', 'petal width (cm)']]
y = iris_df['petal length (cm)']

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
plt.scatter(X_test['sepal length (cm)'], y_test, color='black', label='Actual')
plt.scatter(X_test['sepal length (cm)'], y_pred, color='blue', label='Predicted')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

#In this example, we are using the sepal length and petal width as independent variables to predict the petal length. You can modify the X and y variables based on your specific regression task.

#Make sure to adapt the code according to your specific needs and target variable. Additionally, you might want to explore more advanced regression techniques and feature engineering for a more robust analysis.





