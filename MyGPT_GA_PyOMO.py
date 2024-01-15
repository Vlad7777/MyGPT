#MyGPT_GA_PyOMO

// https://jckantor.github.io/CBE30338/06.04-Linear-Production-Model-in-Pyomo.html


from pyomo.environ import *

# create a model
model = ConcreteModel()

# declare decision variables
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)

# declare objective
model.profit = Objective(expr = 40*model.x + 30*model.y, sense=maximize)

# declare constraints
model.demand = Constraint(expr = model.x <= 40)
model.laborA = Constraint(expr = model.x + model.y <= 80)
model.laborB = Constraint(expr = 2*model.x + model.y <= 100)

# solve
results = SolverFactory('glpk').solve(model)
results.write()
if results.solver.status:
    model.pprint()

# display solution
print('\nProfit = ', model.profit())

print('\nDecision Variables')
print('x = ', model.x())
print('y = ', model.y())

print('\nConstraints')
print('Demand  = ', model.demand())
print('Labor A = ', model.laborA())
print('Labor B = ', model.laborB())







from pyomo.environ import *
import numpy as np

# enter data as numpy arrays
A = np.array([[1, 0], [1, 1],[2,1]])
b = np.array([40, 80,100])
c = np.array([40,30])

# set of row indices
I = range(len(A))

# set of column indices
J = range(len(A.T))

# create a model instance
model = ConcreteModel()

# create x and y variables in the model
model.x = Var(J)

# add model constraints
model.constraints = ConstraintList()
for i in I:
    model.constraints.add(sum(A[i,j]*model.x[j] for j in J) <= b[i])

# add a model objective
model.objective = Objective(expr = sum(c[j]*model.x[j] for j in J), sense=maximize)

# create a solver
solver = SolverFactory('glpk')

# solve
solver.solve(model)

# print solutions
for j in J:
    print("x[", j, "] =", model.x[j].value)

model.constraints.display()
