import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('D:\Data\House1.csv')
#d:\Data
# Split the data into independent and dependent variables
X = data['flow'].values.reshape(-1, 1)
y = data['value'].values.reshape(-1, 1)

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
plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
