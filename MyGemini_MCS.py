#MyGemini_MCS

import numpy as np
import matplotlib.pyplot as plt

def monte_carlo(n):
    # generate n samples from a standard normal distribution
    samples = np.random.normal(0, 1, n)

    # calculate the sample mean
    mean = np.mean(samples)

    # calculate the sample standard deviation
    std_dev = np.std(samples)

    # create a histogram of the samples
    plt.hist(samples)

    # plot a vertical line at the sample mean
    plt.axvline(mean, color='red', linestyle='--', label='Sample mean')

    # plot a vertical line at the standard deviation
    plt.axvline(mean + std_dev, color='green', linestyle='--', label='+1 Standard Deviation')
    plt.axvline(mean - std_dev, color='green', linestyle='--')

    # plot a vertical line at 2 standard deviations
    plt.axvline(mean + 2 * std_dev, color='purple', linestyle='--', label='+2 Standard Deviations')
    plt.axvline(mean - 2 * std_dev, color='purple', linestyle='--')

    # plot a vertical line at 3 standard deviations
    plt.axvline(mean + 3 * std_dev, color='orange', linestyle='--', label='+3 Standard Deviations')
    plt.axvline(mean - 3 * std_dev, color='orange', linestyle='--')

    # set the plot title and labels
    plt.title('Monte Carlo Simulation of Standard Normal Distribution')
    plt.xlabel('Sample')
    plt.ylabel('Number of Samples')

    # show the plot
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # number of samples to generate
    n = 10000

    # run the Monte Carlo simulation
    monte_carlo(n)
