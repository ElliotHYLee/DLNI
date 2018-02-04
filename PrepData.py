import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def rnnData(data, time_steps, isLabel=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
        -> labels == True [3, 4, 5] # labels for predicting the next timestep
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if isLabel:
            rnn_df.append(data.iloc[i + time_steps].as_matrix())
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            if (len(data_.shape) > 1):
                rnn_df.append(data_)
            else:
                rnn_df.append([j] for j in data_)
    return np.array(rnn_df, dtype=np.float32)

def genData():
    pos = np.zeros(4000)
    vel = np.zeros_like(pos)
    for i in range(1, 4000):
        vel[i] = np.sin(i)*i
        pos[i] = pos[i-1] + vel[i]
    x = np.array(vel)
    y = np.array(pos)
    return x, y

def getData():
    x, y = genData()
    y = np.reshape(y, (-1, 1))
    ts = 10
    x = pd.DataFrame(x)
    x = rnnData(x, ts)

    xx = np.zeros((len(x), ts+1))
    yy = np.zeros((len(x), 1))
    for i in range(0, len(x)):
        row = np.reshape(x[i], (-1,))
        initPos = y[i]
        xx[i,:] = np.concatenate((initPos, row), axis=0)
        yy[i] = y[i+ts-1]
    print xx.shape
    print yy.shape

    return xx, yy

if __name__ == '__main__':
    getData()
