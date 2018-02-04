import numpy as np
from PrepData import *
from RNNModel import Model
import tensorflow as tf
import matplotlib.pyplot as plt

def train():
    x, y = getData()

    # print xTrain

    xTrain = x
    yTrain = y

    xTrainforLSTM = np.reshape(xTrain, (xTrain.shape[0], -1, 1))
    yTrainforLSTM = yTrain

    print xTrainforLSTM.shape
    print yTrainforLSTM.shape

    # Train
    model = Model().LSTMModel
    model.fit(xTrainforLSTM, yTrainforLSTM,
              epochs=500, batch_size=2400, validation_split=0.4,
              verbose=1, shuffle=False)

    # # Test
    pred = model.predict(xTrainforLSTM)
    print(pred.shape)
    #print(pred)

    rmse = np.sqrt((np.asarray((np.subtract(pred, yTrainforLSTM))) ** 2).mean())
    print("RSME: %f" % rmse)

    plt.figure()
    plt.plot(yTrainforLSTM, 'r-')
    plt.plot(pred, 'bo-')
    plt.show()

    #print xTrain




if __name__ =='__main__':
    train()
