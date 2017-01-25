


import numpy as np
from matplotlib import pyplot as plt




def roc (N,P):
    # assuming P>N
    xy = (np.hstack((N, P)))
    labelsxy = (np.hstack((np.ones_like(N), np.zeros_like(P))))
    index = np.argsort(xy)[::-1]  # argument of sort in descending order of values thus P > threshold > N
    labelsxy = labelsxy[index]
    FPR = np.cumsum(labelsxy)/len(N)
    TPR = np.cumsum(np.ones_like(labelsxy)-labelsxy)/len(P)
    return FPR, TPR


N = np.random.normal(loc=1, scale=1.0, size=1000)
P = np.hstack ((np.random.normal(loc=1, scale=1.0, size=500) , np.random.normal(loc=3, scale=1.0, size=500)))

print (N)
print (P)

FPR, TPR = roc(N,P)

print FPR


plt.plot(FPR,TPR,'b-', label='orig')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.plot(FPR,TPR + (1-FPR)*0.5,'r-', label='corr')
plt.xlabel('FPR')
plt.ylabel('TPR (corr)')

plt.plot((0,1),(1,1),'r--')


plt.plot(FPR,TPR + (1-FPR)*0.4,'g-', label='under')
plt.xlabel('FPR')
plt.ylabel('TPR (corr)')

plt.plot(FPR,TPR + (1-FPR)*0.6,'m-', label='over')
plt.xlabel('FPR')
plt.ylabel('TPR (corr)')

plt.plot((0,1),(0.5,1),'b--')


plt.legend(loc=4)
plt.axes().set_aspect('equal')
plt.xlim((0,1))
plt.show()


