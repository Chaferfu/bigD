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

populations = [[],[],[],[],[],[],[],[],[],[]]

prototypes = []

train_data, test_data = importData()


print("tri des images")

for i, data in enumerate(train_data['y']):
	if i==500:
		break
	print("image " + str(i))
	if data[0] == 10:
		populations[0].append(train_data['X'][:, :, :, i])
	else:
		populations[data[0]].append(train_data['X'][:, :, :, i])


print(populations[0])

print("Calcul des representants")

for i in range(10):
	print("Calcul du representant de " + str(i))

	populations[i] = np.array(populations[i])
	prototypes.append(np.average(populations[i], axis = 0))


def bigboyRP():

	return




# contraste(train_data["X"])
# image_idx = 0
# print("Label:", train_data["y"][image_idx])
# print(train_data["X"][:, :, :, image_idx])
# plt.imshow(prototypes[0])
# plt.show()

