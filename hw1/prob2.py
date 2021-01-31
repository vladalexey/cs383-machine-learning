import os, sys
import matplotlib
from matplotlib import pyplot as pp
matplotlib.use("TkAgg")
import numpy as np 
from PIL import Image
import cv2
from sklearn.decomposition import PCA
DIR = './yalefaces'

imgs = list()
 
for f in os.listdir(DIR):
    if 'subject' in f:
        img = Image.open(os.path.join(DIR, f))
        imgs.append(np.array(img.resize((40, 40))).flatten())
    
imgs = np.array(imgs)
imgs = (imgs - imgs.mean()) / (imgs.std())

cov_mat = np.cov(imgs.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

pca = imgs.dot(w)

print(eigen_pairs[:2])
print(PCA(2).fit(imgs).components_)

pp.figure(figsize=(20,20))
pp.scatter(pca[:,0], pca[:,1], c='w', edgecolors='r')
pp.show()

