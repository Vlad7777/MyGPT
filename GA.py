#See GA  https://medium.com/@bianshiyao6639/constrained-optimization-using-genetic-algorithm-in-python-958e0139135a

# see https://pypi.org/project/geneticalgorithm2/

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    pen=0
    if X[0]+X[1]<2:
        pen=500+1000*(2-X[0]-X[1])
    return np.sum(X)+pen
    
varbound=[[0,10]]*3

model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

model.run()

