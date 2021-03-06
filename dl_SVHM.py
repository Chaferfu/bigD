from __future__ import print_function
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
torch.manual_seed(0)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        i = 0

        for parameter in self.parameters():
            nbweight = 1
            print(parameter.shape)
            for dim in parameter.shape:
                nbweight *= dim
            i += nbweight
        print(i)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)
            return x

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

            i = 0

            for parameter in self.parameters():
                nbweight = 1
                print(parameter.shape)
                for dim in parameter.shape:
                    nbweight *= dim
                i += nbweight
            print(i)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x), dim=1)

            # print("fct1 has " + str(len(list(self.fc1.parameters())))+"weigths")
            # print(list(self.fc1.parameters()))

           


            return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 90) 
        print("fct1 has " + str(len(list(self.fc1.parameters())))+"weigths")
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(90, 10)  
        # self.fc3 = nn.Linear(500, 10) 

        i = 0

        for parameter in self.parameters():
            nbweight = 1
            print(parameter.shape)
            for dim in parameter.shape:
                nbweight *= dim
            i += nbweight
        print(i) 
    
    def forward(self, x):
            x = x.contiguous().view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc2(x), dim=1)

            return x

def main( model, epoch_nbr = 10, batch_size = 10, learning_rate = 1e-3):

    # Load the dataset
    train_data = loadmat('train_32x32.mat')
    test_data = loadmat('test_32x32.mat')

    train_label = train_data['y'][:2500]
    train_label = np.where(train_label==10, 0, train_label)
    train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)[:2500]

    test_label = test_data['y'][:2500]
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)[:2500]

    # # Hyperparameters
    # epoch_nbr = 15
    # batch_size = 10
    # learning_rate = 1e-3


    if model == "cnn":
        net = CNN()
    elif model == "lenet":
        net = LeNet()
    elif model == "mlp":
        net = MLP()
    else:
        print("erreur : le reseau " + modele + "n'existe pas")
        return

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    trainingTime = 0
    reussiteTrain = ""
    reussiteTest = ""
    resultats = []
    for e in range(epoch_nbr):
        print("Epoch", e)
        startEpoch = time.clock()
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1)
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update

        epochTime = time.clock() - startEpoch

        trainingTime += epochTime

        print("Epoch time : " + str(epochTime))

        startPredictTrain = time.clock()
        predictions_train = net(train_data)
        _, class_predicted = torch.max(predictions_train, 1)
        predictTimeTrain = time.clock() - startPredictTrain
        # print(class_predicted)
        resultTrain = (class_predicted == train_label)
        resultTrain = resultTrain.type(torch.int32)
        reussiteTrain = str(float(sum(resultTrain))/float(len(resultTrain)))
        print("resultTrainaaaaat" + reussiteTrain)

        startPredictTimeTest = time.clock()
        predictions_test = net(test_data)
        _, class_predicted = torch.max(predictions_test, 1)
        predictTimeTest = time.clock() - startPredictTimeTest

        resultTest = (class_predicted == test_label)
        resultTest = resultTest.type(torch.int32)

        reussiteTest = str(float(sum(resultTest))/float(len(resultTest)))
        print("resultTest" + reussiteTest, end="\n\n")

        trainingTimePerEpoch = trainingTime /( e+1)

        resultats.append([str(datetime.datetime.now()), e+1, batch_size, learning_rate, reussiteTrain, reussiteTest, trainingTimePerEpoch, trainingTime , model ])

    return resultats


if __name__ == '__main__':
    main("cnn", epoch_nbr = 10)
