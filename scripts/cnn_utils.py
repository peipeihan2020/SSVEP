"""
Utilities for CNN based SSVEP Classification
"""
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from keras import initializers, regularizers


def CNN_model(input_shape, CNN_PARAMS):
    '''
    Returns the Concolutional Neural Network model for SSVEP classification.

    Args:
        input_shape (numpy.ndarray): shape of input training data
        e.g. [num_training_examples, num_channels, n_fc] or [num_training_examples, num_channels, 2*n_fc].
        CNN_PARAMS (dict): dictionary of parameters used for feature extraction.
        CNN_PARAMS['batch_size'] (int): training mini batch size.
        CNN_PARAMS['epochs'] (int): total number of training epochs/iterations.
        CNN_PARAMS['droprate'] (float): dropout ratio.
        CNN_PARAMS['learning_rate'] (float): model learning rate.
        CNN_PARAMS['lr_decay'] (float): learning rate decay ratio.
        CNN_PARAMS['l2_lambda'] (float): l2 regularization parameter.
        CNN_PARAMS['momentum'] (float): momentum term for stochastic gradient descent optimization.
        CNN_PARAMS['kernel_f'] (int): 1D kernel to operate on conv_1 layer for the SSVEP CNN.
        CNN_PARAMS['n_ch'] (int): number of eeg channels
        CNN_PARAMS['num_classes'] (int): number of SSVEP targets/classes

    Returns:
        (keras.Sequential): CNN model.
    '''

    model = Sequential()
    model.add(Conv2D(2 * CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1),
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))
    model.add(Conv2D(2 * CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']),
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid",
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax',
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))

    return model