# from keras import backend as K
import numpy as np
import os
import sys
from importlib import reload
from utils.data import read_stock_history, date_to_index


# def set_keras_backend(backend):

#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend


def get_history_and_abb():
    # read the data and choose the target stocks for training a toy example
    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = ['AAPL', 'CMCSA', 'REGN']
    training_date_start = '2012-08-13'
    training_date_end = '2015-08-13'  # three years training data
    training_index_start = date_to_index(training_date_start)
    training_index_end = date_to_index(training_date_end)
    target_history = np.empty(shape=(len(target_stocks), training_index_end - training_index_start, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), training_index_start:training_index_end, :]

    # # collect testing data
    # testing_date_start = '2015-08-13'
    # testing_date_end = '2017-08-12'
    # testing_index_start = date_to_index(testing_date_start)
    # testing_index_end = date_to_index(testing_date_end)
    # testing_history = np.empty(shape=(len(target_stocks), testing_index_end - testing_index_start, history.shape[2]))
    # for i, stock in enumerate(target_stocks):
    #     testing_history[i] = history[abbreviation.index(stock), testing_index_start:testing_index_end, :]
    # # normalize
    # for i in range(target_history.shape[0]):
    #     for j in range(target_history.shape[1]):
    #         target_history[i][j] = target_history[i][j]/np.linalg.norm(target_history[i][j])
    # for i in range(testing_history.shape[0]):
    #     for j in range(testing_history.shape[1]):
    #         testing_history[i][j] = testing_history[i][j]/np.linalg.norm(testing_history[i][j])

    return target_history, target_stocks
