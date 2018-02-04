import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, SimpleRNN, Activation, LSTM
from keras import initializers
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU

class Model():
    def __init__(self):
        self.simpleRNNModel = self.getSimpleRNNModel()
        self.LSTMModel = self.getLSTMModel()

    def getLSTMModel(self):
        model = Sequential()
        model.add(LSTM(20,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal',
                  activation=LeakyReLU(),
                  return_sequences = True,
                  input_shape=(11, 1)))
        model.add(Dense(10, kernel_initializer="uniform", activation=LeakyReLU()))
        model.add(Dense(10, kernel_initializer="uniform", activation=LeakyReLU()))
        model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    def getSimpleRNNModel(self):
        model = Sequential()
        model.add(SimpleRNN(11,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation=LeakyReLU(),
                    return_sequences = False,
                    input_shape=(11,1)))
        model.add(Dense(10, kernel_initializer="uniform", activation=LeakyReLU()))
        model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
        #rmsprop = RMSprop(lr=0.001)
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model























#end
