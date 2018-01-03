import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as learn
from tensorflow.contrib import layers as layers
from tensorflow.contrib import rnn
import numpy as np
import tflearn

class Model():
    def __init__(self, opts):
        self.opts = opts
        self.rnn_layers = self.opts.rnn_layers
        self.time_steps = self.opts.time_steps


    def lstm_cells(self):
        return [rnn.DropoutWrapper(
                rnn.BasicLSTMCell(eachLayer['TimeSteps'],
                                  state_is_tuple = True),
                                  eachLayer['keep_prob']
                ) for eachLayer in self.rnn_layers]

    def lstm_model(self):
        self.x = tflearn.input_data([None, self.time_steps, self.opts.inputDim])
        self.x =  tf.unstack(self.x, num=self.time_steps, axis=1)
        self.stacked_lstm = rnn.MultiRNNCell(self.lstm_cells(), state_is_tuple=True)
        self.lstmOutput, _ = rnn.static_rnn(self.stacked_lstm, self.x, dtype=dtypes.float32)

        #self.output = layers.stack(self.output[-1], layers.fully_connected, self.opts.fcDim)
        self.output = tflearn.fully_connected(self.lstmOutput[-1], 10**3)
        self.output = tflearn.fully_connected(self.output, 10**2)
        self.output = tflearn.fully_connected(self.output, 10**1)

        self.output = tflearn.fully_connected(self.output,2)

        self.output = tflearn.regression(self.output,
                                         optimizer='sgd',
                                         metric='R2',
                                         learning_rate= self.opts.learning_rate,
                                         loss='mean_square')
        self.model = tflearn.DNN(self.output)
        return self.model
