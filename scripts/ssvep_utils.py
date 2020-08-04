"""
Utilities for CNN based SSVEP Classification
"""
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import butter, filtfilt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers, regularizers
from sklearn.cross_decomposition import CCA
import sklearn as sk
import numpy as np



def magnitude_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    '''
    
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros(((fft_index_end - fft_index_start), 
                              segmented_data.shape[1], segmented_data.shape[0], 
                              segmented_data.shape[2], segmented_data.shape[3]))
    
    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT)/fft_len
                    magnitude_spectrum = 2*np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[fft_index_start:fft_index_end,]
    
    return features_data

def complex_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    '''
    
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros((2*(fft_index_end - fft_index_start), 
                              segmented_data.shape[1], segmented_data.shape[0], 
                              segmented_data.shape[2], segmented_data.shape[3]))
    
    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT)/fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    features_data[:, channel, target, trial, segment] = np.concatenate((
                        real_part[fft_index_start:fft_index_end,], 
                        imag_part[fft_index_start:fft_index_end,]), axis=0)
    
    return features_data



def find_correlation(n_components, np_buffer, freq):
    cca = CCA(n_components)

    result = np.zeros(freq.shape[0])
    x_coefficients= []
    x_weights = None
    y_weights = None
    xweights=[]
    yweights=[]
    for freq_idx in range(0, freq.shape[0]):
        corr, xscore, xw, yw = find_correlation_for_one_pair(cca, n_components, np_buffer, freq[freq_idx, :, :])
        x_coefficients.append(xscore)
        xweights.append(xw)
        yweights.append(yw)
        max_index = np.argmax(corr)
        result[freq_idx] = corr[max_index]

    freq_max = np.argmax(result)
    scores = x_coefficients[freq_max]
    return freq_max, result[freq_max], scores, xweights[freq_max], yweights[freq_max]

def find_correlation_for_one_pair(cca, n_components, X, Y):
    corr = np.zeros(n_components)
    cca.fit(X.T, np.squeeze(Y).T)

    O1_a, O1_b = cca.transform(X.T, np.squeeze(Y).T)

    for ind_val in range(0, n_components):
        corr[ind_val] = np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1]
    return corr, O1_a, cca.x_weights_, cca.y_weights_



import matplotlib.pyplot as plt
def plot_eeg(data):
    for single in data:
        img = np.zeros(shape=(len(single), len(single)))
        ampl = dict()
        for time in range(len(single)):
            ampl[single[time]] = time
        sorted_ampl = sorted(ampl.items())
        i = 0
        for k in sorted_ampl:
            ind = k[1]
            img[i, 255-ind] = 1
            i += 1
        plt.imshow(img)

