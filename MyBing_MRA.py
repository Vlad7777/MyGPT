#MyBing_MRA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
#data = pd.read_csv('path/to/your/data.csv')

data = pd.read_csv('D:\iris\iris.csv')

#D:\iris\iris.csv
# Split the data into independent and dependent variables
#X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X =  data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species

#y = data['target']
y = data['Species']


# Create a linear regression object
reg = LinearRegression()

# Fit the model using the training data
reg.fit(X, y)

# Print the coefficients
print('Coefficients: \n', reg.coef_)

# Make predictions using the testing set
y_pred = reg.predict(X)

# Plot the results
import matplotlib.pyplot as plt
plt.scatter(y, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
