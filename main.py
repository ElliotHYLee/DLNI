import numpy as np
from PrepData import *
from RNNModel import Model
import tensorflow as tf
import matplotlib.pyplot as plt


def lowPass(vel):
    lp = np.zeros_like(vel)
    lp[0] = vel[0]
    a = 0.9
    for i in range(1, len(vel)):
        lp[i] = a*lp[i-1] + (1-a)*vel[i]
    return lp


def train():
    x, y, x_input, correct = getData()
    print 'ddddd'
    print x_input.shape
    # print xTrain
    r, c = x.shape
    xTrain = x
    yTrain = y

    xTrainforLSTM = np.reshape(xTrain, (xTrain.shape[0], 1, -1))
    yTrainforLSTM = yTrain

    print xTrainforLSTM.shape
    print yTrainforLSTM.shape

    # Train
    model = Model().LSTMModel
    model.fit(xTrainforLSTM, yTrainforLSTM,
              epochs=1000, batch_size=2400, validation_split=0.4,
              verbose=1, shuffle=False)
    # Test
    pred = model.predict(xTrainforLSTM)
    print(pred.shape)
    # print(pred)

    rmse = np.sqrt((np.asarray((np.subtract(pred, yTrainforLSTM))) ** 2).mean())
    print("RSME: %f" % rmse)


    print x_input.shape
    plt.figure()
    plt.plot(correct, 'o-', color='black', label='original signal')
    plt.plot(x_input, '-', color='gray', label='input = original signal + bias + noise')
    plt.plot(pred, 'r-', label='output signal by LSTM')
    plt.plot(lowPass(x_input), 'g-', label='output by low pass (a=0.9)')

    plt.legend(loc='upper left')
    plt.show()

if __name__ =='__main__':
    train()
