#MyGPT_SimulationStatisticsFunctions


"""
Simulating statistical distributions and conducting statistical simulations in Python can be done using various libraries. Here are some common libraries and examples of simulating statistical distributions:

NumPy:
NumPy is a fundamental library for scientific computing in Python and provides functions to generate random samples from various distributions.



"""
import numpy as np
import matplotlib.pyplot as plt

# Simulate random samples from a normal distribution
mu, sigma = 0, 1  # mean and standard deviation
samples = np.random.normal(mu, sigma, 1000)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.5, color='g')
plt.title('Normal Distribution Simulation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

"""
SciPy:
SciPy builds on NumPy and provides additional functionality for scientific computing, including statistical functions.



"""
from scipy.stats import norm

# Simulate random samples from a normal distribution using scipy.stats
mu, sigma = 0, 1
samples = norm.rvs(loc=mu, scale=sigma, size=1000)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.5, color='b')
plt.title('Normal Distribution Simulation (SciPy)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

"""
Statsmodels:
Statsmodels is a library that provides classes and functions for estimating and testing statistical models.
"""



import statsmodels.api as sm

# Simulate a simple linear regression model
np.random.seed(123)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Fit the linear regression model
model = sm.OLS(y, sm.add_constant(X))
results = model.fit()

# Display summary
print(results.summary())

#These are just a few examples, and the libraries mentioned provide a wide range of functions for different statistical simulations. Depending on your specific needs, you might explore other libraries like scikit-learn, simpy, or more domain-specific libraries for finance, physics, etc.


