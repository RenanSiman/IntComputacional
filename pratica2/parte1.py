import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

df = pd.read_csv('data1.csv')
X = df['x'].to_numpy()
Y = df['y'].to_numpy()

alpha = 0.01
m = len(X)
maxEpochs = 1000

# hipotesis function
theta0 = random.random()
theta1 = random.random()
H = theta0 + np.dot(theta1, X)
J = []

for i in range(maxEpochs):     
    error2 = 0
    error = 0
    for j in range(len(X)):
        error = H[j] - Y[j]
        theta0 = theta0 - alpha*(1/m)*error
        theta1 = theta1 - alpha*(1/m)*error*X[j]
        error2 = error2 + error**2
    
    H = theta0 + np.dot(theta1, X)
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
plt.show()