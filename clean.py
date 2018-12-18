#projet de nathan et mathias
# coding: utf-8

from __future__ import print_function
import datetime
import time
import numpy as np
from scipy.io import loadmat
from sklearn import svm
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

	for i in range(len(data)):

		traite = data[:,:,:,i]

		if blackWhite:
			traite = rgb2gray(traite)

		if val != None:
			traite = fondBlanc(traite, val)

		if upper != None and lower != None:
			traite = contraste(traite, upper, lower)

		traitees.append(traite)

	return np.array(traitees);



train_data, test_data = importData()

traitees = pretraitement(train_data["X"], val = 10)

print(traitees)
print(traitees.shape)