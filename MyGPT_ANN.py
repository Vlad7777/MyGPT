#MyGPT_ANN

"""
Creating a basic Artificial Neural Network (ANN) using Python involves using a machine learning library like TensorFlow or PyTorch. In this example, I'll provide a simple Python program using TensorFlow to create and train an ANN for a basic classification task. Make sure to install TensorFlow first:

bash
Copy code
pip install tensorflow
Here's a simple program:

python code
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_clusters_per_class=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=20))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

"""
In this example:

Synthetic data is generated using make_classification from sklearn.datasets.
The data is split into training and testing sets, and standardization is applied using StandardScaler.
A simple ANN model with one hidden layer is created using Sequential from tensorflow.keras.
The model is compiled with the Adam optimizer and binary crossentropy loss for a binary classification task.
The model is trained on the training data for 10 epochs.
Finally, the model is evaluated on the test set.
Note: This is a basic example, and in a real-world scenario, you may need to adjust the architecture, hyperparameters, and data preprocessing based on your specific task and dataset.

"""


