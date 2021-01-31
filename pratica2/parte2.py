import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
#from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import random

df = pd.read_csv('data2.csv')
average = []
maxVFeatures = []
index = df.columns.values
data = df.to_numpy()

for i in index:
    average.append(df[i].mean())
    maxVFeatures.append(df[i].max())


#mean normalization
def meanNormalization(data):
    dt = np.array()
    for col in range(len(index)):
        dt = np.append((data[:,col] - average[col])/maxVFeatures[col])
    return dt

print(meanNormalization(data))

alpha = 0.01
maxEpochs = 1000


# Some tests
""" plt.scatter(data[:,2],data[:,0],c = 'b', marker = '.')
plt.xlabel('House\'s Price (1000*$)')
plt.ylabel('Area (m²)')
plt.show() """

"""
# hipotesis function
theta0 = random.random()
theta1 = random.random()
H = theta0 + np.dot(theta1, X1)
J = []

for i in range(maxEpochs):     
    error2 = 0
    error = 0
    for j in range(len(X1)):
        error = H[j] - Y[j]
        theta0 = theta0 - alpha*(1/m)*error
        theta1 = theta1 - alpha*(1/m)*error*X1[j]
        error2 = error2 + error**2
    
    H = theta0 + np.dot(theta1, X1)
    J.append(1/(2*m)*error2)

# plot graph

# Graph 'J cost' X 'Iterations'
x = np.arange(maxEpochs)
plt.plot(x,J,'g-')
plt.xlabel('Nº of iterations')
plt.ylabel('J cost')
plt.xlim(0, maxEpochs)
plt.show()

# Graph Fit Linear
plt.scatter(X,Y,c = 'b', marker = '.')
plt.plot(X,H,'r-')
plt.xlabel('Population of City in 10.000s')
plt.ylabel('Profit in $10.000')
plt.xlim(4, 24)
plt.ylim(-5, 25)
plt.show() """