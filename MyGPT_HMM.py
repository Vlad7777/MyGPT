#MyGPT_HMM

"""
Hidden Markov Models (HMMs) are a type of statistical model that are particularly useful for modeling time-series data with hidden states. Python provides libraries like hmmlearn and pomegranate for working with Hidden Markov Models. Here, I'll provide a simple example using hmmlearn:

python
Copy code
"""

from hmmlearn import hmm
import numpy as np

# Define the HMM model
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Generate some example data
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 1, (100, 2))])

# Fit the model to the data
model.fit(X)

# Predict hidden states
hidden_states = model.predict(X)

# Print the predicted hidden states
print("Predicted Hidden States:", hidden_states)

"""
In this example:

hmm.GaussianHMM is used to create a Hidden Markov Model with two hidden states (n_components=2) and a full covariance matrix.
X is a synthetic dataset with two normal distributions, one centered at (0, 0) and the other at (5, 5).
The fit method is used to train the model on the data.
The predict method is then used to predict the hidden states of the data.
For a more complex example using the pomegranate library, you can simulate a weather scenario where the weather can be "Sunny" or "Rainy," and you observe whether your friend is either "Happy" or "Sad" based on the weather:

python
Copy code
"""

from pomegranate import *

# Define the HMM model
model = HiddenMarkovModel()

# Define states
sunny = State(DiscreteDistribution({"Happy": 0.8, "Sad": 0.2}), name="Sunny")
rainy = State(DiscreteDistribution({"Happy": 0.4, "Sad": 0.6}), name="Rainy")

# Add states to the model
model.add_states(sunny, rainy)

# Start and end states
model.add_edge(model.start, sunny, 0.5)
model.add_edge(model.start, rainy, 0.5)
model.add_edge(sunny, sunny, 0.8)
model.add_edge(sunny, rainy, 0.2)
model.add_edge(rainy, sunny, 0.3)
model.add_edge(rainy, rainy, 0.7)
model.add_edge(sunny, model.end, 0.1)
model.add_edge(rainy, model.end, 0.1)

# Bake the model
model.bake()

# Generate a sequence of observations
observations = ["Happy", "Sad", "Happy", "Happy", "Sad", "Sad"]

# Predict the most likely sequence of hidden states
predicted_states = model.predict(observations)

# Print the predicted hidden states
print("Predicted Hidden States:", predicted_states)
#In this example, the pomegranate library is used to define a Hidden Markov Model with two states ("Sunny" and "Rainy") and associated emission probabilities. The model is then trained on a sequence of observations ("Happy" and "Sad"), and the most likely sequence of hidden states is predicted.




