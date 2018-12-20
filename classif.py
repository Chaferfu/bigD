import clean
import sys
from scipy.io import loadmat


def importDataFrom(trainFile, testFile):
	train_data = loadmat(trainFile)
	test_data = loadmat(testFile)
	return train_data, test_data;

def main():
	if len(sys.argv) <= 1:
		print("veuillez utiliser les options --train [path/to/train/data] --test [path/to/test/data]")
		return
	else:
		for i in range(len(sys.argv)):
			if sys.argv[i] == "--train":
				if i == len(sys.argv) - 1  or  sys.argv[i+1][0] == "-":
					print(" svp veuillez indiquer  [path/to/train/data]  apres l'argument --train")
				else:
					trainFile = sys.argv[i+1]
			if sys.argv[i] == "--test":
				if i == len(sys.argv) - 1  or  sys.argv[i+1][0] == "-":
					print(" svp veuillez indiquer  [path/to/train/data]  apres l'agrument --train")
				else:
					testFile = sys.argv[i+1]

	train_data, test_data = importDataFrom(trainFile, testFile)
	train_imgs = clean.pretraitement(train_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
	test_imgs = clean.pretraitement(test_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
	clean.KNN(train_imgs, train_data["y"], test_imgs, test_data["y"], 6)

if __name__=="__main__":

	main()