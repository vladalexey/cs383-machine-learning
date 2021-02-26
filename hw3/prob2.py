import numpy as np
import os
import sys
from matplotlib import pyplot as plt

lr = 10e-3
early_stop = 2**(-32)

def init_w(w1, w2):
    return np.array([[w1], [w2]])

def gradient_desc(x1, x2, w1, w2, lr=10e-3, MAX_ITER=10**3):

	N = 1
	it = 0
	prev_loss = None
	change_rate = 1
	w_hist = np.zeros((MAX_ITER, 2))
	cost_hist = np.zeros(MAX_ITER)

	Jsqrt = lambda x1, x2, w1, w2: x1 * w1 - 5 * x2 * w2 - 2

	while change_rate > early_stop and it < MAX_ITER:
		
		jsqrt = Jsqrt(x1, x2, w1, w2)
		prev_loss = jsqrt
		
		w1, w2 = w1 - (1/N) * lr * 2.0 * x1 * jsqrt, w2 - (1/N) * lr * (-10.0) * x2 * jsqrt 
		
		change_rate = (Jsqrt(x1, x2, w1, w2)) ** 2 if not prev_loss else abs((Jsqrt(x1, x2, w1, w2)) ** 2 - prev_loss)
		prev_loss = (Jsqrt(x1, x2, w1, w2)) ** 2

		print("Iter #{}\nTraining loss: {} ".format(it, change_rate))

		w_hist[it, :] = np.array([[w1, w2]], dtype=float)
		cost_hist[it] = (Jsqrt(x1, x2, w1, w2)) ** 2
		it += 1
	
	return [w1, w2], w_hist, cost_hist, it

if __name__ == '__main__':
	
	x1, x2 = 1.0, 1.0
	w1, w2 = 0.0, 0.0
	
	w, hist, cost_hist, it = gradient_desc(x1, x2, w1, w2)
	print(w, cost_hist[-1])
	plt.xlabel("Iteration #")
	plt.ylabel("Loss")
	plt.plot(range(it), cost_hist[:it], color='green', marker='o', linestyle='dashed', linewidth=0.2, markersize=3)

	fig = plt.figure(figsize=(5, 5), dpi=300)
	plot = fig.add_subplot(111, projection='3d')
	plot.yaxis.set_tick_params(labelsize=8)
	plot.xaxis.set_tick_params(labelsize=8)
	plot.zaxis.set_tick_params(labelsize=8)
	plot.set_xlabel("w1")
	plot.set_ylabel("w2")
	plot.set_zlabel("J")
	plot.scatter(hist[:it, 0], hist[:it, 1], cost_hist[:it],
		marker="x", color='r', s=10, linewidth=1, alpha=0.8)
	plot.plot(hist[:it, 0], hist[:it, 1], cost_hist[:it],
		linestyle='dashed', color='b', linewidth=1, alpha=0.8)
	plt.show()