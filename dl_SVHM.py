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

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x), dim=1)

            return x


def main(epoch_nbr = 10, batch_size = 10, learning_rate = 1e-3):

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



    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    trainingTime = 0
    reussiteTrain = ""
    reussiteTest = ""

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

    trainingTimePerEpoch = trainingTime / epoch_nbr

    return [str(datetime.datetime.now()), epoch_nbr, batch_size, learning_rate, reussiteTrain, reussiteTest, trainingTimePerEpoch, trainingTime  ]