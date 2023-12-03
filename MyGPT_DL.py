#MyGPT_DL

"""
Creating a deep learning program in Python often involves using a deep learning library such as TensorFlow or PyTorch. Below is a basic example using TensorFlow and Keras to create a simple deep neural network for image classification. Make sure to install TensorFlow first:

bash
Copy code
pip install tensorflow
Here's an example program:

python
Copy code

"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the deep neural network model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Make predictions on a few test images
predictions = model.predict(test_images[:5])

# Display the predictions
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {tf.argmax(predictions[i])}, Actual: {tf.argmax(test_labels[i])}")
    plt.show()

    """
In this example:

The program uses the MNIST dataset, a dataset of handwritten digits, for image classification.
A simple convolutional neural network (CNN) model is defined using the Keras API, which is part of TensorFlow.
The model is compiled with the Adam optimizer and categorical crossentropy loss for a multi-class classification task.
The model is trained on the training data and evaluated on the test data.
Finally, the model makes predictions on a few test images, and the predictions are displayed along with the actual labels.
Note: This is a basic example, and for more complex tasks, you might need to adjust the architecture, hyperparameters, and data preprocessing based on your specific use case.

"""



