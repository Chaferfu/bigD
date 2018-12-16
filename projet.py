#projet de nathan et mathias
# coding: utf-8

from __future__ import print_function
import datetime
import time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier

def printBandeauNouveauTest(nom):
	s = "☺||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	for i in range(len(nom) + 4):
		s += "|"
	s += "☺"
	print("\n" + s)
	print("☺|||||||||||||||||||||||||||||||  " + nom + "  |||||||||||||||||||||||||||||||☺")
	print(s + "\n")

def printBandeauSimple(nom):
	print("☺|||||||||||||||||||||||||||||||  " + nom + "  |||||||||||||||||||||||||||||||☺\n")

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

def fondBlanc(img):
	av = np.average(img)
	if av < 106 :
		return 255 - img
	else:
		return img

def contraste(img):
	
	img[img>240] = 255
	img[img<=100] = 0
	return img

def CDM_predict(img, prototypes, pretraiter):
	label = 0 
	scores = []
	if pretraiter:
		img = contraste(fondBlanc(rgb2gray(img)))

	img = img.flatten()
	for i in range(len(prototypes)):
		diff = img - prototypes[i] 
		scores.append(sqrt(np.dot(diff,diff)))

	return scores.index(min(scores))

def pretraitement(data):
	print("Application du pretraitement")
	print(data.shape)
	print(len(data))
	imagesRetouchees = []
	for i in range(data.shape[3]):
		#print("ca travaille ", i, data.shape[3], end = '')
		if i % data.shape[3]/100 == 0:
			print("|", end = '')
		imagesRetouchees.append(contraste(fondBlanc(rgb2gray(data[:, :, :, i]))))
	
	#imagesRetouchees = np.array(imagesRetouchees)
	print("")
	return np.array(imagesRetouchees)



def preparerPourFit(data):
	print("Preparation des donnees pour fit le modele")
	if data.shape[-1] > 32:
		data = np.transpose(data, (3,0,1,2))
	print(data.shape , end='')
	nbDim = data.ndim
	nbElt = data.shape[0]
	nbFeatures = 1
	for i in range(1, data.ndim):
		nbFeatures *= data.shape[i]

	data = data.reshape(nbElt, nbFeatures)

	print(" -> ", data.shape)

	return data



def CDM(pretraiter):

	populations = [[],[],[],[],[],[],[],[],[],[]]

	prototypes = []

	train_data, test_data = importData()

	imagesRetouchees = []

	if pretraiter:
		for i in range(len(train_data['y'])):
			imagesRetouchees.append(contraste(fondBlanc(rgb2gray(train_data['X'][:, :, :, i]))))
	else:
		for i in range(len(train_data['y'])):
			imagesRetouchees.append(train_data['X'][:, :, :, i])

	print("tri des images")
	startFit = time.clock()

	for i, data in enumerate(train_data['y']):
		if data[0] == 10:
			populations[0].append(imagesRetouchees[i])
		else:
			populations[data[0]].append(imagesRetouchees[i])


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
	for i, data in enumerate(test_data['y']):
		label = CDM_predict(test_data['X'][:, :, :, i], prototypes, pretraiter)
		# print(data[0], label)
		if data[0] == 10 and label == 0:
			nbTrouve += 1
		elif data[0] == label:
			nbTrouve += 1

	predictTime = str(time.clock() - startPredict)
			
	reussite = str(float(nbTrouve)/float(len(test_data['y'])))
	print("reussitettttttttttttttat" + reussite)

	return [str(datetime.datetime.now()), "yes" if pretraiter else "no", reussite, fitTime, predictTime]


def KNN(pretraiter, n):

	printBandeauNouveauTest("KNN " + ("avec pretraitement" if pretraiter else "sans pretraitement") + " " + str(n) + " voisins")
	printBandeauSimple("Import des données")
	train_data, test_data = importData()

	if pretraiter:
		printBandeauSimple("Pré-traitement des images")
		imagesRetouchees = pretraitement(train_data["X"])
	else:
		imagesRetouchees = train_data["X"]

	X = preparerPourFit(imagesRetouchees)

	y = np.array([])
	for truc in train_data['y']:
		y = np.append(y, (truc[0]))


	neigh = KNeighborsClassifier(n_neighbors=n)

	printBandeauSimple("Entrainement du classifieur")
	startFit = time.clock()
	neigh.fit(X, y)
	fitTime = str(time.clock() - startFit)

	if pretraiter:
		Xtest = preparerPourFit(pretraitement(test_data["X"][:100]))
	else:
		Xtest = preparerPourFit(test_data["X"][:100])

	nbReussites = 0
	printBandeauSimple("Lancement de la prédiction : ")
	startPredict = time.clock()
	reponses = neigh.predict(Xtest[:1000])
	predictTime = str(time.clock() - startPredict)

	for i, reponse in enumerate(reponses):
		print(int(reponse), test_data['y'][i][0])
		if int(reponse) == test_data['y'][i][0]:
			nbReussites+=1

	reussite = str((float(nbReussites)/float(len(reponses))))

	print("Neigh a trouvé " + str(nbReussites) + " reponses justses. Bravo Neigh !")
	print("pourcentage de réussite de Neigh : " + reussite)


	filename = "resultatsKNN.txt"
	with open(filename, mode='a') as file:
		file.write('At %s \t pretraitement = %s\tn=%s\tresult %s\tfit time %s\tpredict time sur 1000 images %s .\n' % 
			(datetime.datetime.now(), "yes" if pretraiter else "no", str(n), reussite, fitTime, predictTime))

	print("REsultats ecris dans " + filename)

	print("au revoir")

	return [str(datetime.datetime.now()), "yes" if pretraiter else "no", str(n), reussite, fitTime, predictTime]







def SVM():


	train_data, test_data = importData()

	X = train_data['X']
	X = np.transpose(X, (3,0,1,2))
	nbElt = X.shape[0]
	nbFeatures = X.shape[1]*X.shape[2]*X.shape[3]
	X = X.reshape(nbElt, nbFeatures)
	print("normalisation !!")
	for i in range(len(X)):
		X[i] = normalize(X[i])
		if i % (len(X)/10):
			print("|", end="")


	print(X[10])

	y = np.array([])
	for truc in train_data['y']:
		y = np.append(y, (truc[0]))
	
	print("shape x", X.shape)
	print(y.shape)
	print("Creation Bob")
	Bob = svm.SVC()
	print("OKI, début du classifieur, on s'entraine hop hop hop")
	Bob.fit(X, y)


	Xtest = test_data['X']
	Xtest = np.transpose(Xtest, (3,0,1,2))
	nbElt = Xtest.shape[0]
	print(nbElt, "C'etrdgklsfgd")
	nbFeatures = Xtest.shape[1]*Xtest.shape[2]*Xtest.shape[3]
	Xtest = Xtest.reshape(nbElt, nbFeatures)

	nbReussites = 0
	print("Bob is predicting, please wait")
	reponses = Bob.predict(Xtest[:100])

	print("Alors maintenant Bob, on va te poser " + str(nbElt) + " questions ok ? ")

	for i, reponse in enumerate(reponses):
		print(int(reponse), test_data['y'][i][0])
		if int(reponse) == test_data['y'][i][0]:
			nbReussites+=1

	print("Bob a trouvé " + str(nbReussites) + " reponses justses. Bravo Bob !")
	reussite = str((float(nbReussites)/float(len(reponses))))
	print("pourcentage de réussite de Bob : " + reussite)
	filename = "resultatsSVM.txt"
	with open(filename, mode='a') as file:
		file.write('At %s result %s .\n' % 
			(datetime.datetime.now(), reussite))

	print("REsultats ecris dans " + filename)
	print("au revoir")


	return 



#SVM()
#CDM()
# for i in range(1,15):
# 	KNN(True, i)

# for i in range(1,15):
# 	KNN(False, i)

