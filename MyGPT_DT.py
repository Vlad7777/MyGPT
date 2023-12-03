#MyGPT_DT

"""

Decision Trees are a popular machine learning algorithm used for both classification and regression tasks. In Python, you can use the scikit-learn library to create and work with Decision Trees. Here's a simple example for a classification task:

python
Copy code

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (a well-known dataset for classification)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

"""

In this example:

We use the Iris dataset, which is included in scikit-learn.
The dataset is split into training and testing sets using train_test_split.
A Decision Tree classifier is created using DecisionTreeClassifier.
The classifier is trained on the training data using the fit method.
Predictions are made on the test data using the predict method.
The accuracy of the classifier is evaluated using accuracy_score.
If you want to visualize the decision tree, you can use the plot_tree function from sklearn.tree:

python
Copy code

"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the decision tree
plt.figure(figsize=(10, 7))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

"""
This will generate a visual representation of the Decision Tree.

Make sure to have scikit-learn installed:

bash
Copy code

"""

#pip install scikit-learn
#This is a basic example, and you can further customize and tune the Decision Tree model based on your specific requirements.




