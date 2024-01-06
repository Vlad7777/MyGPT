#Convex optimization
# see  https://habr.com/ru/companies/ods/articles/448054/


from scipy.optimize import Bounds, minimize
import numpy as np
import matplotlib.pyplot as plt

labels =      ['Акции', 'Облигации', 'Недвижимость', 'Золото']
mu = np.array([  10.9,     5.2,        10.8,        7.0])

var = np.array([15.2,      3.6,        19.2,        15.6])

R = np.array( [[1.,        0.,         0.59,        0.04],
              [0.,         1.0,        0.19,        0.28],
              [0.59,       0.19,       1.,          0.13],
              [0.04,       0.28,       0.13,        1.]])

var = np.expand_dims(var, axis=0)
S = var.T @ var * R
# Initial guess
x = np.ones(4) * 0.25

def value(x):
        return x.T @ S @ x

def optimize_portfolio(r):
    mu_cons = {'type': 'eq',
                 'fun': lambda x: np.sum(mu @ x.T) - r
                }
    sum_cons = {'type': 'eq',
                 'fun': lambda x: np.sum(x) - 1
                }
    bnds = Bounds (np.zeros_like(x), np.ones_like(x) * np.inf)

    res = minimize(value, x, method='SLSQP', 
                   constraints=[mu_cons, sum_cons], bounds=bnds)
    return res.x

rate = np.arange(4, 12, 0.1)
y = np.array(list(map(optimize_portfolio, rate))).T

plt.figure(figsize=(16, 6))
plt.stackplot(rate, y, labels=labels)
plt.legend(loc='upper left')
plt.show()
