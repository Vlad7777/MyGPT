#MyGPT_InventoryTheory

"""
Inventory theory involves managing the stock of goods to meet customer demand while minimizing costs. Python can be used for various aspects of inventory management, including demand forecasting, reorder point calculation, and safety stock optimization. Below is a simple example of how you can perform basic inventory calculations using Python.

Let's consider the classic Economic Order Quantity (EOQ) model, which helps determine the optimal order quantity that minimizes the total inventory holding and ordering costs.

python
Copy code
"""

import math

def calculate_eoq(demand, setup_cost, holding_cost):
    """
    Calculate Economic Order Quantity (EOQ) using the EOQ formula.
    
    Parameters:
        demand (float): Annual demand for the product.
        setup_cost (float): Setup (ordering) cost per order.
        holding_cost (float): Holding (carrying) cost per unit per year.

    Returns:
        float: Optimal order quantity (EOQ).
    """
    eoq = math.sqrt((2 * demand * setup_cost) / holding_cost)
    return eoq

def calculate_total_cost(eoq, demand, setup_cost, holding_cost):
    """
    Calculate the total cost associated with the EOQ model.
    
    Parameters:
        eoq (float): Economic Order Quantity.
        demand (float): Annual demand for the product.
        setup_cost (float): Setup (ordering) cost per order.
        holding_cost (float): Holding (carrying) cost per unit per year.

    Returns:
        float: Total cost.
    """
    order_cost = (demand / eoq) * setup_cost
    holding_cost = (eoq / 2) * holding_cost
    total_cost = order_cost + holding_cost
    return total_cost

# Example usage
demand = 1000  # Annual demand
setup_cost = 50  # Setup (ordering) cost per order
holding_cost = 2  # Holding (carrying) cost per unit per year

# Calculate EOQ
eoq = calculate_eoq(demand, setup_cost, holding_cost)
print(f"Optimal Order Quantity (EOQ): {eoq:.2f} units")

# Calculate total cost associated with the EOQ model
total_cost = calculate_total_cost(eoq, demand, setup_cost, holding_cost)
print(f"Total Cost at EOQ: ${total_cost:.2f}")

"""
In this example:

calculate_eoq calculates the Economic Order Quantity (EOQ) using the EOQ formula.
calculate_total_cost computes the total cost associated with the EOQ model, considering both ordering and holding costs.
You can adapt this example and explore more sophisticated inventory management techniques based on your specific needs and data. Additionally, libraries like NumPy and pandas can be used for data manipulation and analysis in more complex inventory scenarios.


"""

