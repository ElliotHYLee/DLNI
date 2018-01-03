import numpy as np
import tensorflow as tf

class Opts():
    def __init__(self):
        self.log_dir = 'logs/'
        self.time_steps = 10
        self.inputDim = 11
        self.rnn_layers = [{'TimeSteps': self.time_steps, 'keep_prob': 1},
                           #{'TimeSteps': self.time_steps, 'keep_prob': 1},
                           {'TimeSteps': self.time_steps, 'keep_prob': 1}]
        #RNN_LAYERS = [{'TimeSteps': TIMESTEPS, 'keep_prob': 1}]
        self.exp = 3
        self.fcDim = np.multiply(np.ones(5), 10**self.exp).astype(int).tolist()
        self.batch_size = 100
        self.lr = 0.01
        self.learning_rate = tf.train.exponential_decay(
                                learning_rate = self.lr,
                                global_step = tf.Variable(0),
                                decay_steps = 100,
                                decay_rate = 0.9,
                                staircase = False,
                                name = None)
