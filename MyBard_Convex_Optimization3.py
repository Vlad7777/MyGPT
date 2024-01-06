#MyBard_Convex_Optimization3

"""
Convex optimization with constraints in Python can be achieved using various libraries and techniques. Here's a summary of two popular approaches:

Using SciPy.optimize: SciPy.optimize is a built-in Python library that provides functions for solving convex optimization problems. It offers a variety of algorithms for different types of convex problems, including quadratic programming (QP), second-order cone programming (SOCP), and semidefinite programming (SDP).

Using Pyomo: Pyomo is an open-source Python modeling language and solver toolkit specifically designed for convex optimization. It provides a clear and concise modeling language, allowing you to define convex optimization problems declaratively. Pyomo also integrates with external solvers, such as CPLEX and Gurobi, which can efficiently solve large-scale convex optimization problems.

Here's an example of how to use SciPy.optimize to solve a simple quadratic programming (QP) problem:

"""

import numpy as np
from scipy.optimize import minimize

#my add
x=1


def objective_function(x):
    return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)

# Define the problem data
Q = np.array([[2, 1], [1, 3]])  # Hessian matrix of the objective function
c = np.array([2, 1])  # Gradient vector of the objective function
A = np.array([[1, 1]])  # Matrix of constraint coefficients
b = np.array([3])  # Vector of constraint values

# Solve the QP problem
res = minimize(objective_function, x0=np.zeros(2), method='SLSQP', args=(Q, c), constraints={'type': 'eq', 'fun': lambda x: np.dot(A, x) - b})

# Print the solution
print('Solution:')
print(res.x)




"""
Use code with caution. Learn more
This code will solve the QP problem of minimizing the objective function 0.5 * (2x1^2 + 3x2^2) + 2x1 + x2 subject to the constraint x1 + x2 = 3. The SLSQP method is used to solve the QP problem, and the constraints parameter is used to specify the constraint function.

Here's an example of how to use Pyomo to solve a more complex SDP problem:

"""

import pyomo.environ as pyo

model = pyo.ConcreteModel()

x1 = pyo.Var(initialize=0)  # Variable representing x1
x2 = pyo.Var(initialize=0)  # Variable representing x2

# Define the objective function
model.objective = pyo.Objective(expr=np.dot(c, x), sense=pyo.maximize)

# Add the constraints
model.constraint1 = pyo.Constraint(expr=x1 + x2 == 3)
model.constraint2 = pyo.Constraint(expr=2 * x1 + 3 * x2 <= 15)

solver = pyo.SolverFactory('cplex')
solver.solve(model)

print('Solution:')
print(model.x1())
print(model.x2())

#This code will solve the SDP problem of maximizing the objective function 2x1 + 3x2 subject to the constraints x1 + x2 = 3 and 2x1 + 3x2 <= 15. The cplex solver will be used to solve the SDP problem efficiently.


