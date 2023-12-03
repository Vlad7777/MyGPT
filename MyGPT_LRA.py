#MyGPT_LRA

"""
Certainly! Logistic Regression is a popular algorithm for binary classification problems. Below is an example Python program using the scikit-learn library to implement Logistic Regression on a sample dataset. Make sure to install scikit-learn first:

bash
Copy code
pip install scikit-learn
Here's a basic example:

python
Copy code

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the iris dataset (for binary classification)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Consider only the first two features for simplicity
y = (iris.target != 0) * 1  # Convert to binary classification problem (setosa or not)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Plot decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Logistic Regression Decision Boundary')
plt.show()

"""
In this example:

We use the Iris dataset for binary classification by considering only the first two features and converting the target to binary (setosa or not).
The dataset is split into training and testing sets.
Logistic Regression model is created, trained, and evaluated on the test set.
Decision boundary is plotted on the feature space.
Note: This is a basic example, and for a real-world scenario, you might need to handle data preprocessing, handle multiclass classification, and tune hyperparameters based on your specific dataset and problem.


"""