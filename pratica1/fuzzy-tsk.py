# Aluno: Renan Siman Claudino - Matrícula: 201522040420 
## Function f(x) = x²
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from math import exp

## Generate 100 numbers in vector X
X = np.around(np.arange(-5,5,0.1),1)
#print(X)
## 

a = 0.01 # alpha
errorEpochs = []
fit = 1
maxEpochs = 1000

## Inicial Parameters
p1 = -2
p2 = 2
q1 = 0
q2 = 0
x1m = -2
x2m = 2
sigma1 = 1
sigma2 = 1

index = np.random.permutation(len(X))   
for j in range(0,maxEpochs):

    ## Permute indexes of vector X
    
    ## Quadratic error
    error2 = 0 

    for i in index:
        error = 0

        for j in range(0,fit):
            w1 = exp(-((X[i]-x1m)/sigma1)**2/2)
            w2 = exp(-((X[i]-x2m)/sigma2)**2/2)

            y1 = p1*X[i] + q1
            y2 = p2*X[i] + q2
            y = (w1*y1 + w2*y2)/(w1+w2)

            f_Xi = X[i]**2
            # print("f(x) = ",f_Xi)
            # print("yE = ", y)
            error = y - f_Xi
            # print("Erro = ",error)
            
            w1d = w1/(w1+w2)
            w2d = w2/(w1+w2)
            
            ## Fit parameters using gradient
            p1 = p1 - a*error*X[i]*w1d
            p2 = p2 - a*error*X[i]*w2d

            q1 = q1 - a*error*w1d
            q2 = q2 - a*error*w2d

            x1m = x1m - a*error*w2*((y1-y2)/(w1+w2)**2)*w1*(X[i]-x1m)/(sigma1**2)
            x2m = x2m - a*error*w1*((y2-y1)/(w1+w2)**2)*w2*(X[i]-x1m)/(sigma2**2)

            sigma1 = sigma1 - a*error*w2*((y1-y2)/((w1+w2)**2))*((X[i]-x1m)**2/(sigma1**3))
            sigma2 = sigma2 - a*error*w1*((y2-y1)/((w1+w2)**2))*((X[i]-x1m)**2/(sigma2**3))

        error2 = error**2/2

    errorEpochs.append(error2)

print("Erro da última época: ", errorEpochs[maxEpochs-1])
yGrad = []

for i in range(len(X)):
    w1 = exp(-((X[i]-x1m)/sigma1)**2/2)
    w2 = exp(-((X[i]-x1m)/sigma2)**2/2)
    
    y1 = p1*X[i] + q1
    y2 = p2*X[i] + q2
    yGrad.append((w1*y1 + w2*y2)/(w1+w2))

## Graph
fig, ax = plt.subplots()
ax.plot(X, X**2, '-b', label='f(x)=x²')
ax.plot(X,yGrad, '-r', label='Função usando método do gradiente')
ax.legend(['f(x)=x²','yGrad'])
plt.ylabel('Axis Y')
plt.xlabel('Axis X')
plt.title('Gráfico: f(x)=x²  X  yGrad')
plt.show()