README.txt

classif.py utilise le modele KNN de scikit learn et applique au prealable notre pretraitement sur les donnees de test et d'entrainement

Pour l'utiliser : 
au minimum

python classif.py --test [fichier.mat contenant les donnees de test]

vous pouvez utiliser les options --train pour inquiquer un fichier contenant des donnees sur lesquelles vous souhaitez entrainer le modele et/ou --nbTest pour indiquer un nombre d'images a tester 

par exemple, les commandes suivantes fonctionnent

python classif.py --test test_32x32.mat --nbTest 100 --train train_32x32.mat
python classif.py --test test_32x32.mat --nbTest 100

Le temps de prediction peut etre élevé si on essaie de predire un trop grand set d'images, nous vous recommandons d'en tester environ 100 si vous ne voulez pas attendre plus de quelques dizaines secondes