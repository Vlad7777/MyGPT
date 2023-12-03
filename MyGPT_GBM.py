#MyGPT_GBM

"""
Gradient Boosting Machines (GBM) is a machine learning technique that builds a series of decision trees and combines their predictions to improve accuracy and reduce overfitting. There are several libraries in Python that implement GBM, and one popular one is scikit-learn. Here's a simple example of using GBM for a regression task:

python
Copy code
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
data = load_boston()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting Regressor
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Feature importance
feature_importance = gbm.feature_importances_

# Plot feature importance
plt.bar(range(len(feature_importance)), feature_importance, tick_label=data.feature_names)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in GBM')
plt.show()

"""
In this example:

We use the Boston Housing dataset from scikit-learn for a regression task.
The dataset is split into training and testing sets.
The GradientBoostingRegressor is used to create and train the GBM model.
Predictions are made on the test set, and the mean squared error is calculated.
Feature importance is plotted to show the contribution of each feature to the model.
Make sure to have scikit-learn installed:

bash
Copy code
pip install scikit-learn
You can customize this example based on your specific regression task and dataset. GBM is a versatile algorithm that is also applicable to classification tasks. You can adjust hyperparameters like the number of estimators, learning rate, and maximum depth to optimize the model for your particular problem.


"""

