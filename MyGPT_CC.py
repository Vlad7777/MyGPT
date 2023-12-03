#MyGPT_CC

"""
Control charts, also known as Shewhart charts or process-behavior charts, are a statistical tool used to monitor the stability and consistency of a process over time. In Python, you can create control charts using various libraries, and one popular choice is the matplotlib library for visualization and statistical calculations.

Here's a simple example of how to create a basic control chart using Python with matplotlib:

python
Copy code

"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate some random data for demonstration purposes
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=50)

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Define control limits (e.g., using 3-sigma limits)
upper_limit = mean + 3 * std_dev
lower_limit = mean - 3 * std_dev

# Create time axis
time_axis = np.arange(1, len(data) + 1)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time_axis, data, label='Data')

# Plot the mean line
plt.axhline(mean, color='r', linestyle='--', label='Mean')

# Plot the control limits
plt.axhline(upper_limit, color='g', linestyle='--', label='Upper Control Limit')
plt.axhline(lower_limit, color='g', linestyle='--', label='Lower Control Limit')

# Highlight points outside the control limits
out_of_control = np.where((data > upper_limit) | (data < lower_limit))[0]
plt.scatter(time_axis[out_of_control], data[out_of_control], color='red', label='Out of Control')

# Add labels and legend
plt.title('Control Chart')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()

# Show the plot
plt.show()

"""
In this example:

The red dashed line represents the mean of the data.
The green dashed lines represent the upper and lower control limits (commonly set at 3 standard deviations from the mean).
Points outside the control limits are considered "out of control."
This is a basic example, and you might need to adapt it based on the specific requirements of your data and the type of control chart you want to create (e.g., X-bar charts for monitoring the mean, R charts for monitoring the range).

If you are interested in a more comprehensive statistical analysis, you might consider using libraries like statsmodels or control for more advanced control chart capabilities.


"""

