#MyBard_Convex_Optimization 

"""
Convex optimization with constraints in Python can be achieved using various libraries and techniques. Here's a summary of two popular approaches:

Using SciPy.optimize: SciPy.optimize is a built-in Python library that includes functions for solving convex optimization problems. It provides the minimize_constrained function, which takes a convex objective function, a linear constraint matrix, and a vector of right-hand side values.

Using Pyomo: Pyomo is an open-source modeling language and solver toolkit specifically designed for convex optimization problems with constraints. It provides a clear and concise modeling language, allowing you to define convex optimization problems declaratively. Pyomo also integrates with external solvers, such as CPLEX and Gurobi, which can efficiently solve large-scale convex optimization problems.

Here's an example of how to use SciPy.optimize to solve a simple quadratic programming (QP) problem:


"""

import numpy as np
from scipy.optimize import minimize_constrained

# Define the problem data
c = np.array([2, 3])  # Objective function coefficients
Q = np.array([[4, 2], [2, 5]])  # Quadratic objective matrix
A = np.array([[1, 2], [3, 1]])  # Coefficients of the linear constraints
b = np.array([5, 4])  # Right-hand side values of the constraints

# Solve the QP problem
res = minimize_constrained(lambda x: np.dot(x.T, np.dot(Q, x)) + np.dot(c.T, x),
                          np.zeros(2),
                          A,
                          b,
                          constraints=lambda x: np.dot(A, x) <= b)

# Print the solution
print('Solution:')
print(res.x)

"""
Use code with caution. Learn more
This code will solve the QP problem of minimizing the quadratic form 2x1^2 + 3x2^2 subject to the constraints x1 + 2x2 <= 5 and 3x1 + x2 <= 4. The minimize_constrained function will return the optimal values for x1 and x2.

Here's an example of how to use Pyomo to solve a more complex convex optimization problem:

"""
import pyomo.environ as pyo

model = pyo.ConcreteModel()

x1 = pyo.Var(initialize=0)  # Variable representing the amount of product A to produce
x2 = pyo.Var(initialize=0)  # Variable representing the amount of product B to produce

# Define the objective function
model.objective = pyo.Objective(expr=50 * x1 + 60 * x2, sense=pyo.maximize)

# Define the constraints
model.labor_hours = pyo.Constraint(expr=2 * x1 + 3 * x2 <= 100)
model.demand_A = pyo.Constraint(expr=x1 <= 20)
model.demand_B = pyo.Constraint(expr=x2 <= 30)

solver = pyo.SolverFactory('cplex')
status = solver.solve(model)

if status == pyo.SolverStatus.ok:
    print('Solution:')
    print(model.x1())
    print(model.x2())
else:
    print('Solver failed to find a solution:')
    print(model.solution_status)

    
#This code will solve the convex optimization problem of maximizing the profit of a company that produces two products, A and B, given the constraints from the previous example. The cplex solver will be used to solve the convex optimization problem efficiently.


