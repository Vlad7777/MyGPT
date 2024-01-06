#MyBard_Linear programming

"""

Python offers various libraries and tools for optimizing linear programming (LP) models. Here's a summary of two popular approaches:

Using SciPy.optimize: SciPy.optimize is a built-in Python library that includes functions for solving LP problems. It provides the linprog function, which takes a matrix of coefficients, a vector of objective function coefficients, and a vector of right-hand side values.

Using Pyomo: Pyomo is an open-source modeling language and solver toolkit specifically designed for LP problems. It provides a clear and concise modeling language, allowing you to define LP problems declaratively. Pyomo also integrates with external solvers, such as CPLEX and Gurobi, which can efficiently solve large-scale LP problems.

Here's an example of how to use SciPy.optimize to solve a simple LP problem:


"""

import numpy as np
from scipy.optimize import linprog

# Define the problem data
c = np.array([2, 3])  # Objective function coefficients
A = np.array([[1, 2], [3, 1]])  # Coefficients of the constraints
b = np.array([5, 4])  # Right-hand side values of the constraints

# Solve the LP problem
res = linprog(c, A, b)

# Print the solution
print('Solution:')
print(res.x)

"""
Use code with caution. Learn more
This code will solve the LP problem of maximizing the objective function 2x1 + 3x2 subject to the constraints x1 + 2x2 <= 5 and 3x1 + x2 <= 4. The linprog function will return the optimal values for x1 and x2.

Here's an example of how to use Pyomo to solve a more complex LP problem:


"""
#import CPLEX
import pyomo.environ as pyo

model = pyo.ConcreteModel()

x1 = pyo.Var(initialize=0)  # Variable representing the amount of product A to produce
x2 = pyo.Var(initialize=0)  # Variable representing the amount of product B to produce

# Define the objective function
model.objective = pyo.Objective(expr=50 * x1 + 60 * x2, sense=pyo.maximize)

# Add the constraints
model.labor_hours = pyo.Constraint(expr=2 * x1 + 3 * x2 <= 100)
model.demand_A = pyo.Constraint(expr=x1 <= 20)
model.demand_B = pyo.Constraint(expr=x2 <= 30)

solver = pyo.SolverFactory('cplex')
solver.solve(model)

print('Profit:', model.objective())
print('x1:', model.x1())
print('x2:', model.x2())


#This code will solve the LP problem of maximizing the profit of a company that produces two products, A and B, given the constraints from the previous example. The cplex solver will be used to solve the LP problem efficiently.

