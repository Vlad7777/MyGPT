#MyGPT_UL
"""
Unsupervised learning is a type of machine learning where the algorithm is given data without explicit instructions on what to do with it. Common tasks in unsupervised learning include clustering, dimensionality reduction, and density estimation. Here are examples using Python and popular libraries like scikit-learn:

1. Clustering (K-Means):
Clustering is the process of grouping similar data points together.
python
Copy code

"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=0, scale=1, size=(100, 2)),
                       np.random.normal(loc=5, scale=1, size=(100, 2))])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-Means Clustering')
plt.show()

"""
2. Dimensionality Reduction (Principal Component Analysis - PCA):
PCA is used to reduce the number of features in a dataset while preserving its variance.
python
Copy code

"""

from sklearn.decomposition import PCA

# Generate synthetic data
np.random.seed(42)
data = np.random.rand(100, 3)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Visualize the reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.title('PCA for Dimensionality Reduction')
plt.show()

"""
3. Density Estimation (Gaussian Mixture Model - GMM):
GMM is used for modeling the distribution of data.
python
Copy code

"""


from sklearn.mixture import GaussianMixture

# Generate synthetic data
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=0, scale=1, size=(100, 1)),
                       np.random.normal(loc=5, scale=1, size=(100, 1))])

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)
gmm.fit(data)

# Visualize the density estimation
plt.scatter(data, np.zeros_like(data), c=gmm.predict(data), cmap='viridis')
plt.title('Gaussian Mixture Model for Density Estimation')
plt.show()

"""
These are just basic examples, and depending on your specific use case, you might need to adjust parameters and choose different algorithms. Always refer to the documentation of the libraries and algorithms for more details:

scikit-learn documentation: https://scikit-learn.org/stable/documentation.html

"""


