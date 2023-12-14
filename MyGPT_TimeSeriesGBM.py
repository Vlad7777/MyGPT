#MyGPT_TimeSeriesGBM

#Geometric Brownian Motion (GBM) is often used to model the behavior of financial instruments such as stock prices. In Python, you can simulate a GBM process using the numpy library. Here's an example:


import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(T, mu, sigma, S0, dt):
    """
    Simulate a geometric Brownian motion process.

    Parameters:
    - T: Time horizon in years.
    - mu: Drift (average rate of return per unit time).
    - sigma: Volatility (standard deviation of the rate of return).
    - S0: Initial value of the process.
    - dt: Time step size.

    Returns:
    - t: Array of time points.
    - S: Array of simulated prices.
    """
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)
    W = np.random.standard_normal(size=n-1)
    W = np.insert(W, 0, 0.0)
    W = np.cumsum(W) * np.sqrt(dt)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S

# Parameters
T = 1.0  # Time horizon in years
mu = 0.1  # Drift
sigma = 0.2  # Volatility
S0 = 100.0  # Initial value
dt = 1/252  # Time step size (assuming 252 trading days in a year)

# Simulate GBM
t, S = geometric_brownian_motion(T, mu, sigma, S0, dt)

# Plot the simulation
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='GBM Simulation')
plt.title('Geometric Brownian Motion (GBM) Simulation')
plt.xlabel('Time (Years)')
plt.ylabel('Price')
plt.legend()
plt.show()

"""
In this example:

geometric_brownian_motion is a function that simulates a GBM process given the time horizon T, drift mu, volatility sigma, initial value S0, and time step size dt.

The parameters T, mu, sigma, S0, and dt are set to values suitable for a financial time series simulation.

The simulated GBM process is then plotted using Matplotlib.

Feel free to adjust the parameters to match your specific scenario or use real financial data if you have it. Note that this is a basic simulation, and in practice, GBM may not perfectly capture all aspects of real financial markets.



"""
