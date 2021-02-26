import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib
from matplotlib import pyplot as pp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from collections import defaultdict
from sklearn.decomposition import PCA

matplotlib.use('TKAgg')
np.set_printoptions(threshold=sys.maxsize)
DIR = './diabetes.csv'
THRESH_HOLD = 2**(-23)
MAX_ITER = 10**4
COLORS = ["g", "r", "c", "b", "k", "m", "y"]

def L2_dist(a, b):
	return (sum((a - b)**2))**0.5


def visualize(centroids, classifications, iter):

	fig = pp.figure(figsize=(5, 5), dpi=300)s
	plot = fig.add_subplot(111, projection='3d')
	plot.view_init(30, 10 * iter)
	for idx, centroid in enumerate(centroids):
		color = COLORS[idx]
		plot.scatter(centroids[centroid][0], centroids[centroid][1], centroids[centroid][2],
					 edgecolors="k", marker="o", color=color, s=80, linewidths=1.5)

	for idx, classification in enumerate(classifications):
		color = COLORS[idx]
		for featureset in classifications[classification]:
			plot.scatter(featureset[0], featureset[1], featureset[2],
						 marker="x", color=color, s=10, linewidths=1, alpha=0.5)

	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	img = cv2.resize(img, (720, 720))
	return img

def calc_purity(origin, DATA_LENGTH, centroids, pca ):
	purity = 0

	for x_test in origin.values():

		labels_count = defaultdict(int)

		x_test = np.array(x_test).astype(float)
		x_test = pca.transform(x_test)

		for feature in x_test:
			distances = [(centroid_idx, L2_dist(feature, centroid))
							for centroid_idx, centroid in centroids.items()]
			# get the idx of the closer centroid to the current datapoint
			distances.sort(key=lambda x: x[1])
			labels_count[distances[0][0]] += 1
		print(labels_count)
	
		purity += max(labels_count.values())

	print(purity/DATA_LENGTH)
	return purity/DATA_LENGTH


def myKMeans(data, origin, DATA_LENGTH, pca, k: int):
	np.random.seed(955432)
	idxs = np.random.choice(DATA_LENGTH, k, replace=False)
	video = cv2.VideoWriter(
		'K_{}.avi'.format(k), cv2.VideoWriter_fourcc(*'MJPG'), 1,  (720, 720))

	centroids = {idx: data[idx] for idx in idxs}

	for iter in range(MAX_ITER):

		classifications = {}

		for centroid_idx in centroids.keys():
			classifications[centroid_idx] = []

		for featureset in data:
			# calculate distance of the current datapoint to K centroids
			distances = [(centroid_idx, L2_dist(featureset, centroid))
						 for centroid_idx, centroid in centroids.items()]
			# get the idx of the closer centroid to the current datapoint
			distances.sort(key=lambda x: x[1])
			classifications[distances[0][0]].append(featureset)

		prev_centroids = dict(centroids)

		for classification in classifications:
			centroids[classification] = np.average(classifications[classification], axis=0)

		optimized = True

		for c in centroids:
			original_centroid = prev_centroids[c]
			current_centroid = centroids[c]
			print(np.sum(np.abs((current_centroid - original_centroid))))
			if np.sum(np.abs((current_centroid - original_centroid))) > THRESH_HOLD:
				optimized = False

		frame = visualize(centroids, classifications, iter)
		frame = cv2.putText(frame, "Iteration #" + str(iter), (frame.shape[1] // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA) 

		purity = calc_purity(origin, DATA_LENGTH, centroids, pca)

		frame = cv2.putText(frame, "Purity " + str(round(purity, 4)), (frame.shape[1] // 2 - 100, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA) 
		video.write(frame)

		if optimized:
			break
		
	video.release()

if __name__ == "__main__":

	data = defaultdict(list)
	x = list()
	y = list()
	f = open(DIR, 'r')

	for line in f.readlines():
		ins = [float(c) for c in line.strip().split(',')]
		y.append(ins[0])
		x.append(ins[1:])
		data[ins[0]].append(ins[1:])
	DATA_LENGTH = len(x)

	x = np.array(x)
	y = np.array(y)

	if len(x[1]) > 3:
		pca = PCA(3).fit(x)
		reduced = pca.transform(x)
	else:
		reduced = x

	for k in (np.random.choice(range(1, 8), 5)):
		myKMeans(reduced, data, DATA_LENGTH, pca, k)
	


	
	