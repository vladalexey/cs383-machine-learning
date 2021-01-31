import os, sys
import numpy as np 
import matplotlib.pyplot as plt 

x = [[0, 1],
    [0, 0],
    [1, 1],
    [0, 0],
    [1, 1],
    [1, 0],
    [1, 0],
    [1, 1],
    [2, 0],
    [2, 1]]

y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

x = np.array(x)
y = np.array(y).reshape(1, -1)

x = (x - x.mean()) / x.std()

cov = np.cov(x.T)
vals, vecs = np.linalg.eigh(cov)

pairs = [(np.absolute(vals[i]), vecs[:, i]) for i in range(len(vals))]
pairs.sort(key=lambda k: k[0], reverse=False)
w = np.hstack((pairs[0][1][:, np.newaxis], pairs[1][1][:, np.newaxis]))
x = np.dot(x, w)

plt.scatter(x[:, 0], x[:, 1], c="w", edgecolors="r")
plt.show()
