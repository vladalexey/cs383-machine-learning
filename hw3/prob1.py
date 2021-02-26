import numpy as np 

x = np.array([[-2, -5, -3,  0, -8, -2,  1,  5, -1,  6], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 2)
y = np.array([[1, -4, 1, 3, 11, 5, 0, -1, -3, 1]]).reshape(-1, 1)

w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print("Weight:", w)

preds = np.dot(x, w)
print("Predictions:", preds)

import math
rmse = math.sqrt(np.square(np.subtract(y,preds)).mean())
print("RMSE:",rmse)
