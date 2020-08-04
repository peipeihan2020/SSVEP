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
from scripts import preprocess as pre
import stimuli as st
from scipy import signal



all_acc = list()
sample_rate = 256

flicker_freq = st.frequencies
channels = 13

def combine_training():
    path= '../data/subjects'
    subject = np.zeros((len(st.frequencies),15, channels, sample_rate*4))
    for fr in range(0,len(st.frequencies)):
        for block in range(0, 15):
            filename=os.path.join(path, str(fr)+'_'+str(block)+ '.npz')
            data =np.load(filename)['data'].T
            re_data = signal.resample(data, sample_rate*4).T
            subject[fr, block, :, :] = re_data
    target = subject.shape[0]
    trial = subject.shape[1]
    channel= subject.shape[2]
    sample = subject.shape[3]
    subject = subject.reshape((target, channel, sample, trial))
    return subject

def get_data(eeg, filter='butter'):
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
    re_data = signal.resample(eeg.T, sample_rate).T
    filtered_data = pre.bandpass_filter(re_data, 6, 80, sample_rate, 4, filter)


    return filtered_data

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


def get_templates(filter='buffer'):
    segmented_data = combine_training()
    segmented_data = pre.get_filtered_eeg(segmented_data, 6, 80, 4, sample_rate, filter, 1)
    template_signals = []
    for target in range(0, segmented_data.shape[0]):
        data = segmented_data[target, :, :, :]
        part = np.mean(data,axis=2)
        template_signals.append(part)
    template =  np.array(template_signals)
    #template = template.resize((template.shape[0], template.shape[1], template.shape[2], 1))

    return template

