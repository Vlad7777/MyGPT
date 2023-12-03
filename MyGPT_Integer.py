#MyGPT_Integer

"""
Integer Programming is a type of mathematical optimization where some or all of the decision variables are required to be integers. A popular Python library for solving Integer Programming problems is PuLP. Below is an example of solving an Integer Programming problem using PuLP:

Let's consider an example of a simple production planning problem:

python
Copy code
"""

from pulp import LpProblem, LpMaximize, LpVariable, lpSum

# Create a maximization problem
production_problem = LpProblem("Production_Planning", LpMaximize)

# Decision variables (integer)
x1 = LpVariable("x1", lowBound=0, cat='Integer')  # units of Product 1
x2 = LpVariable("x2", lowBound=0, cat='Integer')  # units of Product 2

# Objective function
production_problem += 5 * x1 + 4 * x2, "Total_Profit"

# Constraints
production_problem += 2 * x1 + x2 <= 8, "Resource_Constraint_1"
production_problem += x1 + 2 * x2 <= 6, "Resource_Constraint_2"

# Solve the problem
production_problem.solve()

# Print the results
#print("Status:", LpProblem.status[production_problem.status])

print("Optimal Solution:")
for var in production_problem.variables():
    print(f"{var.name}: {var.varValue}")
print("Total Profit:", lpSum([5 * x1.varValue, 4 * x2.varValue]))

"""
In this example:

We create a maximization problem using LpProblem.
Decision variables x1 and x2 are defined as integers.
An objective function and constraints are added to the problem.
The solve() method is called to find the optimal solution.
Make sure to install PuLP before running the code:

bash
Copy code
pip install pulp
This is a simple example, and Integer Programming can be applied to more complex problems with additional constraints and decision variables. Adjust the code based on the specific requirements of your Integer Programming problem.

"""
