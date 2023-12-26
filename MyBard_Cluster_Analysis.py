#Cluster_Analysis

"""
Here's an overview of cluster analysis in Python:

Purpose:

To group similar data points together based on their features, forming clusters that reveal underlying patterns or relationships in the data.
Techniques:
K-means
Hierarchical clustering
Density-based clustering
Key Libraries:

scikit-learn: Comprehensive machine learning library with various clustering algorithms.
NumPy: Fundamental library for numerical computations and array manipulation.
pandas: Data analysis and manipulation library, often used for loading and preprocessing data.
Matplotlib or Seaborn: Data visualization libraries for visualizing clusters and results.
General Steps:

Load and Preprocess Data:
Import data (e.g., from CSV, Excel, or other formats).
Handle missing values and outliers.
Normalize or standardize features if necessary.
Choose a Clustering Algorithm:
Select based on data characteristics and desired outcomes.
Create a Cluster Model:
Instantiate the chosen algorithm with appropriate parameters.
Fit the model to the data.
Predict Cluster Labels:
Assign each data point to a cluster.
Evaluate Results:
Visualize clusters (e.g., scatter plots, heatmaps).
Use metrics like silhouette scores or cohesion/separation scores.
Specific Examples:


K-means Clustering:
"""

import numpy as np
from sklearn.cluster import KMeans

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Create a K-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(X)

# Predict cluster labels for each data point
labels = kmeans.predict(X)

# Print cluster labels
print(labels)


#Hierarchical Clustering:


from sklearn.cluster import AgglomerativeClustering

# Create a hierarchical clustering model
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Fit the model to the data
hc.fit(X)

# Predict cluster labels
labels = hc.labels_

# Print cluster labels
print(labels)

