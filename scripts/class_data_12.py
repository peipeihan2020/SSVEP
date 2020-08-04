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
import sys
import os
sys.path.insert(0, os.path.abspath(''))
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import preprocess as pre

data_path = os.path.abspath('../data')

all_acc = list()
sample_rate = 256

flicker_freq = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75])

def get_data(window_len, shift_len, filter='butter'):
    '''
        Returns segmented data by window length and shift length.

        Args:
            window_len (float): length of window. (s)
            shift_len (float): length of shift window (s).

        Returns:
            (numpy.ndarray): segmented data.
        '''
    all_segment_data = dict()
    all_filtered_data = dict()
    for subject in np.arange(0, 10):
        path = '{data_path}\s{subject}.mat'.format(data_path=data_path, subject=subject+1)
        dataset = sio.loadmat(path)
        eeg = np.array(dataset['eeg'], dtype='float32')

        filtered_data = pre.get_filtered_eeg(eeg, 6, 80, 4, sample_rate, filter)
        all_filtered_data[subject+1] = filtered_data
        all_segment_data[subject + 1] = pre.get_segmented_epochs(filtered_data, window_len,
                                                                      shift_len, sample_rate)

    return all_segment_data, all_filtered_data

def _get_cca_reference_signals(data_len, target_freq, harmonics):
    '''
        Returns target array for one target.

        Args:
            data_len (float): number of samples.
            target_freq (float): frequency of stimuli target (Hz).

        Returns:
            (numpy.ndarray): target signals with harmonics.
    '''
    reference_signals = []
    for i in range(1, harmonics+1, 1):
        t = np.arange(0, (data_len / (sample_rate)), step=1.0 / (sample_rate))
        reference_signals.append(np.sin(np.pi * 2 * i * target_freq * t))
        reference_signals.append(np.cos(np.pi * 2 * i * target_freq * t))

    reference_signals = np.array(reference_signals)

    return reference_signals


def get_targets(window_len, harmonics):
    '''
            Returns all target frequencies.

            Args:
                window_len (float): length of window. (s)

            Returns:
                (numpy.ndarray): target signals with harmonics.
    '''
    duration = int(window_len * sample_rate)
    reference_templates = []
    for fr in range(0, len(flicker_freq)):
        reference_templates.append(_get_cca_reference_signals(duration, flicker_freq[fr], harmonics))
    reference_templates = np.array(reference_templates, dtype='float32')
    return reference_templates


def _get_template_signals(segmented_data, total_cycles):
    template_signals = []
    for target in range(0, segmented_data.shape[0]):
        data = segmented_data[target, :, :, :total_cycles]
        part = np.mean(data,axis=2)
        template_signals.append(part)
    return np.array(template_signals)

def get_templates(filteredData, window_len, shift_len):
    reference_templates = dict()
    for subject in filteredData.keys():
        template = filteredData[subject]
        template = _get_template_signals(template, 14)
        template.resize((template.shape[0], template.shape[1], template.shape[2], 1))
        reference_templates[subject] = pre.get_segmented_epochs(template, window_len,
                                                                   shift_len, sample_rate)

    return reference_templates
