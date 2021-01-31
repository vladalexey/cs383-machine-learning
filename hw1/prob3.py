import os, sys
import numpy as np 
from PIL import Image
import cv2

DIR = './yalefaces'
imgs = list()
 
for f in os.listdir(DIR):
    if 'subject' in f:
        img = Image.open(os.path.join(DIR, f))
        imgs.append(np.array(img.resize((40, 40))).flatten())

origin = Image.open(os.path.join(DIR, 'subject02.centerlight')).resize((40, 40))
origin = np.array(origin).flatten()
origin = origin.reshape(1, -1)

imgs = np.array(imgs)
mean = imgs.mean()
std = imgs.std()
imgs = (imgs - mean) / std
origin = (origin - mean ) / std

cov_mat = np.cov(imgs.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

video = cv2.VideoWriter('video3.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10,  (40, 40))

for k in range(1, len(eigen_pairs)):
    w = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(k)])
    pca = np.dot(origin, w)

    revert = np.dot(pca, w.T)
    revert = revert * std + mean

    revert = np.reshape(revert, (40, 40))
    final = np.uint8(np.absolute(revert))

    video.write(cv2.cvtColor(final, cv2.COLOR_GRAY2BGR))

cv2.destroyAllWindows()
video.release()
