#projet de nathan et mathias
# coding: utf-8

from __future__ import print_function
import datetime
import cPickle
import time
import numpy as np
from scipy.io import loadmat
from sklearn import svm
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def importData():
	train_data = loadmat("train_32x32.mat")
	test_data = loadmat("test_32x32.mat")
	return train_data, test_data;

def fondBlanc(img, val = 106):
	av = np.average(img)
	if av < val :
		return 255 - img
	else:
		return img

def contraste(img, upper = 240, lower = 100):
	
	img[img>upper] = 255
	img[img<=lower] = 0
	return img

def pretraitement(data, val = None, upper = None, lower = None, blackWhite = None):

	traitees = []

	data = np.transpose(data, (3,0,1,2))

	nbImages = len(data)

	for i in range(nbImages):

		traite = data[i,:,:,:]

		if blackWhite:
			traite = rgb2gray(traite)

		if val != None:
			traite = fondBlanc(traite, val)

		if upper != None and lower != None:
			traite = contraste(traite, upper, lower)

		traitees.append(traite)

	traitees = np.array(traitees)
	
	if blackWhite:
		traitees = traitees.reshape(nbImages, 32*32)
	else:
		traitees = traitees.reshape(nbImages, 32*32*3)

	return np.array(traitees);

def acp(data, nc):

	pca = PCA(n_components = nc)
	data = pca.fit_transform(data)

	return data

def CDM_predict(img, prototypes):

	label = 0 
	scores = []
	
	for i in range(len(prototypes)):
		diff = img - prototypes[i] 
		scores.append(sqrt(np.dot(diff,diff)))

	return scores.index(min(scores))

def CDM(train_imgs, train_labels, test_imgs, test_labels):

	populations = [[],[],[],[],[],[],[],[],[],[]]

	prototypes = []

	print("tri des images")
	startFit = time.clock()

	for i, data in enumerate(train_labels):
		if data[0] == 10:
			populations[0].append(train_imgs[i])
		else:
			populations[data[0]].append(train_imgs[i])


	print("Calcul des representants")
	for i in range(10):
		print("Calcul du representant de " + str(i) + " ")

		populations[i] = np.array(populations[i])
		prototypes.append(np.average(populations[i], axis = 0))
		prototypes[i] = prototypes[i].flatten()

	fitTime = str(time.clock() - startFit)

	print("prediction")
	startPredict = time.clock()
	nbTrouve = 0
	for i, data in enumerate(test_labels):
		label = CDM_predict(test_imgs[i], prototypes)
		if data[0] == 10 and label == 0:
			nbTrouve += 1
		elif data[0] == label:
			nbTrouve += 1

	predictTime = str(time.clock() - startPredict)
			
	reussite = str(float(nbTrouve)/float(len(test_labels)))
	print("reussitettttttttttttttat" + reussite)

	return [str(datetime.datetime.now()), reussite, fitTime, predictTime]

def KNN(train_imgs, train_labels, test_imgs, test_labels, n):
	print("KNN "  + str(n) + " voisins")

	y = np.array([])
	for truc in train_labels:
		y = np.append(y, (truc[0]))


	neigh = KNeighborsClassifier(n_neighbors=n)

	print("Entrainement du classifieur")
	print(train_imgs.shape, y.shape)
	startFit = time.clock()
	neigh.fit(train_imgs, y)
	fitTime = str(time.clock() - startFit)

	nbReussites = 0
	print("Lancement de la prédiction : ")
	startPredict = time.clock()
	reponses = neigh.predict(test_imgs)
	predictTime = str(time.clock() - startPredict)

	for i, reponse in enumerate(reponses):
		if int(reponse) == test_labels[i][0]:
			nbReussites+=1

	reussite = str((float(nbReussites)/float(len(reponses))))

	print("Neigh a trouvé " + str(nbReussites) + " reponses justses. Bravo Neigh !")
	print("pourcentage de réussite de Neigh : " + reussite)

	print("au revoir")

	return [str(datetime.datetime.now()), str(n), reussite, fitTime, predictTime]

def saveTrainedKNN(train_imgs, train_labels, n, fileName):

	print("KNN "  + str(n) + " voisins")

	y = np.array([])
	for truc in train_labels:
		y = np.append(y, (truc[0]))


	neigh = KNeighborsClassifier(n_neighbors=n)

	print("Entrainement du classifieur")
	neigh.fit(train_imgs, y)
	print("fini.")

	with open(fileName, 'wb') as fid:
		cPickle.dump(neigh, fid)

	print("classifieur sauvegardé dans le fichier " + fileName)

def trainedKNN(test_imgs, test_labels, fileName):

	print("Import du modele KNN entrainé")
	with open(fileName, 'rb') as fid:
		neigh = cPickle.load(fid)

	nbReussites = 0
	print("Lancement de la prédiction : ")
	reponses = neigh.predict(test_imgs)

	for i, reponse in enumerate(reponses):
		if int(reponse) == test_labels[i][0]:
			nbReussites+=1

	reussite = str((float(nbReussites)/float(len(reponses))))

	print("Neigh a trouvé " + str(nbReussites) + " reponses justses. Bravo Neigh !")
	print("pourcentage de réussite de Neigh : " + reussite)

	print("au revoir")

	print("PS : voici les reponses que neigh a trouve")
	print(reponses)

	return reponses



if __name__ == "__main__":
	
	print("Import des donnees" + str(time.clock()))
	train_data, test_data = importData()

	print("pretraitement" + str(time.clock()))
	traitees = pretraitement(train_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
	traitessTest = pretraitement(test_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)

	# traitees = acp(traitees, 500)
	# traitessTest = acp(traitessTest, 500)

	trainedKNN(traitessTest[:100], test_data["y"][:100], "myKNN.pkl")