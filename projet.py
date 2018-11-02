#projet de nathan et mathias
# coding: utf-8

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt



def importData():
	train_data = loadmat("train_32x32.mat")
	test_data = loadmat("test_32x32.mat")
	return train_data, test_data;

def contraste(img):
	
	img[img>155] = 255
	img[img<=20] = 0


		
h





train_data, test_data = importData();

contraste(train_data["X"])
image_idx = 0
print("Label:", train_data["y"][image_idx])
print(train_data["X"][:, :, :, image_idx])
plt.imshow(train_data["X"][:, :, :, image_idx])
plt.show()




