#projet de nathan et mathias
# coding: utf-8

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm


def importData():
	train_data = loadmat("train_32x32.mat")
	test_data = loadmat("test_32x32.mat")
	return train_data, test_data;

def contraste(img):
	
	img[img>155] = 255
	img[img<=20] = 0

def CDM():

	populations = [[],[],[],[],[],[],[],[],[],[]]

	prototypes = []

	train_data, test_data = importData()


	print("tri des images")

	for i, data in enumerate(train_data['y']):
		if data[0] == 10:
			populations[0].append(train_data['X'][:, :, :, i])
		else:
			populations[data[0]].append(train_data['X'][:, :, :, i])


	print("Calcul des representants")

	for i in range(10):
		print("Calcul du representant de " + str(i) + " ")

		populations[i] = np.array(populations[i])
		prototypes.append(np.average(populations[i], axis = 0))






	# contraste(train_data["X"])
	# image_idx = 0
	# print("Label:", train_data["y"][image_idx])
	# print(train_data["X"][:, :, :, image_idx])
	plt.imshow(prototypes[1])
	plt.show()

	return

def SVM():
	train_data, test_data = importData()

	X = train_data['X']
	X = np.transpose(X, (3,0,1,2))
	nbElt = X.shape[0]
	nbFeatures = X.shape[1]*X.shape[2]*X.shape[3]
	X = X.reshape(nbElt, nbFeatures)

	y = np.array([])
	for truc in train_data['y']:
		y = np.append(y, (truc[0]))
	
	print(X.shape)
	print(y.shape)
	print("Creation Bob")
	Bob = svm.SVC(verbose = True)
	print("OKI, début du classifieur, on s'entraine hop hop hop")
	Bob.fit(X[:100], y[:100])

	print("Alors maintenant Bob, on va te poser " + str(len(test_data)) + " questions ok ? ")

	Xtest = test_data['X']
	Xtest = np.transpose(Xtest, (3,0,1,2))
	nbElt = Xtest.shape[0]
	nbFeatures = Xtest.shape[1]*Xtest.shape[2]*Xtest.shape[3]
	Xtest = Xtest.reshape(nbElt, nbFeatures)

	nbReussites = 0
	reponses = Bob.predict(Xtest)
	for i, reponse in enumerate(reponses):
		print(int(reponse), test_data['y'][i][0])
		if int(reponse) == test_data['y'][i][0]:
			nbReussites+=1

	print("Bob a trouvé " + str(nbReussites) + " reponses justses. Bravo Bob !")
	print("pourcentage de réussite de Bob : " + str((nbReussites/len(reponses))))
	print("au revoir")

SVM()