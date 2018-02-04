import numpy as np
from PrepData import *
from RNNModel import Model
import tensorflow as tf
import matplotlib.pyplot as plt

def train():
    x, y = getData()

    print 'im here 1'

    xTrain = x['train']
    yTrain = y['train']
    xVal = x['val']
    yVal = y['val']

    # print xTrain

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    xTrainforRNN = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    yTrainforRNN = yTrain
    #yTrainforRNN = np.reshape(yTrain, (yTrain.shape[0], yTrain.shape[1], 1)) #return_sequences = true

    xTrainforLSTM = np.reshape(xTrain, (xTrain.shape[0], 1, xTrain.shape[1]))
    yTrainforLSTM = yTrain
    #yTrainforLSTM = np.reshape(yTrain, (yTrain.shape[0], 1, yTrain.shape[1]))#return_sequences = true
    print xTrain.shape
    print yTrain.shape

    # Train
    model = Model().LSTMModel
    model.fit(xTrainforLSTM, yTrainforLSTM, epochs=1000, batch_size=73, verbose=1,  shuffle=False, validation_split=0.2)

    print 'im here 2'
    # Test
    pred = model.predict(xTrainforLSTM)
    print(pred.shape)
    #print(pred)

    rmse = np.sqrt((np.asarray((np.subtract(pred, yTrainforLSTM))) ** 2).mean())
    print("RSME: %f" % rmse)

    plt.figure()
    plt.plot(yTrainforLSTM, 'ro')
    plt.plot(pred, 'b')
    plt.show()


if __name__ =='__main__':
    train()
