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
    pos = np.zeros(100)
    vel = np.zeros_like(pos)
    for i in range(1, 100):
        vel[i] = i
        pos[i] = pos[i-1] + vel[i]

    x = np.array(vel)
    y = np.array(pos)

    return x, y

def splitData(data, val_size=0.1, test_size=0.1):
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
    return df_train, df_val, df_test

def getData():
    x, y = genData()
    y = np.reshape(y, (len(y), 1))
    ts = 10
    #z = np.zeros(ts)
    #x = np.concatenate( (z, x), axis = 0)
    x = pd.DataFrame(x)
    x = rnnData(x, ts)

    xx = np.zeros((len(x), ts+1))
    yy = np.zeros((len(x),1))
    for i in range(0, len(x)):
        row = np.reshape(x[i], (10,))
        initPos = y[i]
        xx[i,:] = np.concatenate((initPos, row), axis=0)
        yy[i] = y[i+9]
    # print xx.shape
    # print yy.shape

    x = pd.DataFrame(xx)
    y = pd.DataFrame(yy)

    xTrain, xVal, xTest = splitData(x)
    yTrain, yVal, yTest = splitData(y)

    xx = dict(train = xTrain, val = xVal, test = xTest)
    yy = dict(train = yTrain, val = yVal, test = yTest)

    return xx, yy


if __name__ == '__main__':
    getData()
