#MyGPT_Cluster

"""
Cluster analysis and k-nearest neighbors (KNN) are two distinct techniques, often used for different purposes. Let me provide you with examples for both:

Cluster Analysis using K-Means:
Cluster analysis is used to group similar data points into clusters. K-Means is a popular clustering algorithm. Here's an example using scikit-learn:

python
Copy code
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create synthetic data with three clusters
data, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
plt.title('K-Means Clustering')
plt.show()

"""
In this example:

make_blobs generates synthetic data with three clusters.
KMeans is applied to cluster the data into three groups.
The clusters and cluster centers are visualized using matplotlib.
K-Nearest Neighbors (KNN) Classification:
K-Nearest Neighbors is a supervised learning algorithm used for classification and regression. Here's an example using scikit-learn:

python
Copy code
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Apply K-Nearest Neighbors classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

"""
In this example:

The Iris dataset is split into training and testing sets.
KNeighborsClassifier is applied to classify the data.
Predictions are made on the test data, and accuracy is calculated.
Make sure to have scikit-learn installed:

bash
Copy code
pip install scikit-learn
Adjust the examples based on your specific data and requirements. Clustering and KNN serve different purposes, so choose the technique that fits your problem.
"""