import os
import sys
from dtw import dtw, accelerated_dtw
import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from scripts import ssvep_utils as su

sys.path.insert(0, os.path.abspath('..'))
data_path = os.path.abspath('../data')
all_segment_data = dict()
all_acc = list()
window_len = 1
shift_len = 1
sample_rate = 256
duration = int(window_len*sample_rate)
flicker_freq = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75])


def get_cca_reference_signals(data_len, target_freq, sampling_rate):
    reference_signals = []
    t = np.arange(0, (data_len / (sampling_rate)), step=1.0 / (sampling_rate))
    reference_signals.append(np.sin(np.pi * 2 * 1 * target_freq * t))
    reference_signals.append(np.cos(np.pi * 2 * 1 * target_freq * t))
    reference_signals.append(np.sin(np.pi * 2 * 2 * target_freq * t))
    reference_signals.append(np.cos(np.pi * 2 * 2 * target_freq * t))
    reference_signals.append(np.sin(np.pi * 2 * 3 * target_freq * t))
    reference_signals.append(np.cos(np.pi * 2 * 3 * target_freq * t))
    # reference_signals.append(np.sin(np.pi * 2 * 4 * target_freq * t))
    # reference_signals.append(np.cos(np.pi * 2 * 4 * target_freq * t))

    reference_signals = np.array(reference_signals)

    return reference_signals

def get_tacca_template_signals(data_len, sampling_rate):
    reference_signals = []
    t = np.arange(0, (data_len / (sampling_rate)), step=1.0 / (sampling_rate))
    phases = [0, np.pi/2, (3./4)*np.pi, np.pi, (5/4)*np.pi, (3/2)*np.pi]
    for target_freq in flicker_freq:
        freq_signals=[]
        for phase in phases:
            phase_signals =[]
            phase_signals.append(np.sin(np.pi * 2 * 1 * target_freq * t+phase))
            phase_signals.append(np.cos(np.pi * 2 * 1 * target_freq * t + phase))
            phase_signals.append(np.sin(np.pi * 2 * 2 * target_freq * t + phase))
            phase_signals.append(np.cos(np.pi * 2 * 2 * target_freq * t + phase))
            phase_signals.append(np.sin(np.pi * 2 * 3 * target_freq * t + phase))
            phase_signals.append(np.cos(np.pi * 2 * 3 * target_freq * t + phase))
            # phase_signals.append(np.sin(np.pi * 2 * 4 * target_freq * t + phase))
            # phase_signals.append(np.cos(np.pi * 2 * 4 * target_freq * t + phase))
            freq_signals.append(phase_signals)
        reference_signals.append(freq_signals)
    return reference_signals


def get_dtw(fre,X, xweights, yweights):
    # data_length = segmented_data.shape[1]
    # weights = weights.T
    # segmented_data = segmented_data.T
    # X =np.repeat(weights,data_length,axis=0)
    # # segmented_data = np.multiply(X, segmented_data).T
    # segmented_data = np.dot(segmented_data, weights.T)
    manhattan_distance = lambda x, y: np.abs(x - y)
    X = np.dot(X.T, xweights)

    distances = []
    for template in fre:

        template = np.array(template).T
        Y = np.dot(template, yweights)
        d, cost_matrix, acc_cost_matrix, path = dtw(np.squeeze(X), np.squeeze(Y), dist=manhattan_distance)
        distances.append(d)
    distances = preprocessing.minmax_scale(distances)
    return min(distances)

import eeg_plot as pl

def tacca_classify(segmented_data, reference_templates, reference_tacca_templates):
    predicted_class = []
    labels = []
    cca = CCA(1)
    index = 1
    for target in range(0, segmented_data.shape[0]):
        for trial in range(0, segmented_data.shape[2]):
            for segment in range(0, segmented_data.shape[3]):
                pl.plot(segmented_data[target, :, trial, segment, :], target, index)
                index += 1
    #             labels.append(target)
    #             results = []
    #             for t in range(0, segmented_data.shape[0]):
    #                 result, coe, xweights, yweights  = su.find_correlation_for_one_pair(cca, 1, segmented_data[target, :, trial, segment, :],
    #                                                                    reference_templates[t, :, :])
    #                 distance = get_dtw(reference_tacca_templates[t], segmented_data[target, :, trial, segment, :],xweights,yweights)
    #                 results.append(result-distance)
    #             result = np.argmax(results)+1
    #             # if result == target or predict_class == target:
    #             #     predict_class = target
    #             print('cca {result}; '.format(result=result))
    #
    #             predicted_class.append(result)
    # labels = np.array(labels) + 1
    # predicted_class = np.array(predicted_class)

    # return labels, predicted_class




for subject in np.arange(0, 10):
    path = '{data_path}\s{subject}.mat'.format(data_path=data_path, subject=subject+1)
    dataset = sio.loadmat(path)
    eeg = np.array(dataset['eeg'], dtype='float32')

    num_classes = eeg.shape[0]
    n_ch = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]

    filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
    all_segment_data[subject + 1] = su.get_segmented_epochs(filtered_data, window_len,
                                                                  shift_len, sample_rate)

reference_templates = []
for fr in range(0, len(flicker_freq)):
    reference_templates.append(get_cca_reference_signals(duration, flicker_freq[fr], sample_rate))
reference_templates = np.array(reference_templates, dtype='float32')

reference_tacca_templates = get_tacca_template_signals(duration, sample_rate)

for subject in all_segment_data.keys():
    labels, predicted_class = tacca_classify(all_segment_data[subject], reference_templates, reference_tacca_templates)
    c_mat = confusion_matrix(labels, predicted_class)
    accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
    all_acc.append(accuracy)
    print('Subject: {subject}, Accuracy: {accuracy} %'.format(subject=subject, accuracy = accuracy*100))


all_acc = np.array(all_acc)
print('Overall Accuracy Across Subjects: {all_acc} %, std: {std} %'.format(all_acc=np.mean(all_acc)*100, std=np.std(all_acc)*100))