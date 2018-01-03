import numpy as np
from PrepData import *
from Opts import Opts
from RNNModel import Model

def train():
    x, y = getData()
    options = Opts()
    # Get Model
    model = Model(options).lstm_model()

    # Train
    model.fit(x['train'], y['train'],
              validation_set = (x['val'], y['val']),
              show_metric=True,
              batch_size=100,
              n_epoch = 10)

    # Test
    pred = model.predict( x['test'])
    print(pred.shape)
    print(pred)

    rmse = np.sqrt((np.asarray((np.subtract(pred, y['test']))) ** 2).mean())
    score = mean_squared_error(pred, y['test'])
    nmse = score / np.var(y['test']) # should be variance of original data and not data from fitted model, worth to double check
    print("RSME: %f" % rmse)
    print("NSME: %f" % nmse)
    print("MSE: %f" % score)
















if __name__ =='__main__':
    train()
