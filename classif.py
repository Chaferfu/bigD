import clean
import sys
from scipy.io import loadmat


def importDataFrom(file):
	data = loadmat(file)
	return data

def main():
	if len(sys.argv) <= 1:
		print("veuillez utiliser les options --train [path/to/train/data] --test [path/to/test/data]")
		return
	else:
		trainFile = ""
		nbTest = None
		for i in range(len(sys.argv)):
			if sys.argv[i] == "--train":
				if i == len(sys.argv) - 1  or  sys.argv[i+1][0] == "-":
					print(" svp veuillez indiquer  [path/to/train/data]  apres l'argument --train")
					return
				else:
					trainFile = sys.argv[i+1]
			if sys.argv[i] == "--test":
				if i == len(sys.argv) - 1  or  sys.argv[i+1][0] == "-":
					print(" svp veuillez indiquer  [path/to/train/data]  apres l'agrument --train")
					return
				else:
					testFile = sys.argv[i+1]

			if sys.argv[i] == "--nbTest":
				if i == len(sys.argv) - 1  or  sys.argv[i+1][0] == "-":
					print(" svp veuillez indiquer  le nombre de tests  apres l'agrument --nbTest")
					return
				else:
					nbTest = int(sys.argv[i+1])


	if trainFile == "":
		print("Lancement du classifieur KNN contenu dans le fichier myKNN.pkl")
		test_data = importDataFrom(testFile)
		test_imgs = clean.pretraitement(test_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
		clean.trainedKNN(test_imgs[:nbTest], test_data["y"][:nbTest], "myKNN.pkl")
		


	else:
		train_data = importDataFrom(trainFile)
		test_data = importDataFrom(testFile)
		train_imgs = clean.pretraitement(train_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
		test_imgs = clean.pretraitement(test_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
		clean.KNN(train_imgs, train_data["y"], test_imgs[:nbTest], test_data["y"][:nbTest], 6)


if __name__=="__main__":

	main()

