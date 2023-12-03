#MyGPT_NPV

"""
Net Present Value (NPV) is a financial metric that represents the difference between the present value of cash inflows and outflows over a period of time. It is commonly used in capital budgeting to assess the profitability of an investment.

Here's a simple example of how to calculate NPV in Python:

python
Copy code
"""

def calculate_npv(rate, cashflows):
    npv = 0
    for i, cashflow in enumerate(cashflows):
        npv += cashflow / (1 + rate) ** (i + 1)
    return npv

# Example cashflows: initial investment followed by annual cash inflows
cashflows = [-1000, 300, 300, 300, 300]

# Example discount rate (as a decimal)
discount_rate = 0.1

# Calculate NPV
npv_result = calculate_npv(discount_rate, cashflows)

# Print the result
print(f'Net Present Value (NPV): {npv_result:.2f}')

"""
In this example:

cashflows represent the cash inflows and outflows over a series of periods. The first value is typically a negative value representing the initial investment, and subsequent values are positive representing cash inflows.
discount_rate is the discount rate used to calculate the present value of future cashflows.
The calculate_npv function iterates through each cashflow, discounts it to present value, and sums up the results.
Note: This is a basic example, and in real-world scenarios, you might want to use more sophisticated techniques, and tools like NumPy can be helpful for vectorized operations to improve efficiency.

Here's an example using NumPy:

python
Copy code
"""

import numpy as np

def calculate_npv(rate, cashflows):
    periods = np.arange(1, len(cashflows) + 1)
    npv = np.sum(cashflows / (1 + rate) ** periods)
    return npv

# Example cashflows and discount rate
cashflows = np.array([-1000, 300, 300, 300, 300])
discount_rate = 0.1

# Calculate NPV
npv_result = calculate_npv(discount_rate, cashflows)

# Print the result
print(f'Net Present Value (NPV): {npv_result:.2f}')
#This NumPy version is more concise and efficient, especially when dealing with larger datasets.





