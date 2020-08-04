import os
import sys
from dtw import dtw, accelerated_dtw
import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
import math

from scripts import ssvep_utils as su


def tacca_classify(segmented_data, templates, total_cycles):
    predicted_class = []
    labels = []
    for target in range(0, segmented_data.shape[0]):
        for trial in range(total_cycles, segmented_data.shape[2]):
            for segment in range(0, segmented_data.shape[3]):
                labels.append(target)
                refe = templates[:, :, :, segment, :]
                result,*_= su.find_correlation(1, segmented_data[target, :, trial, segment, :],refe)
                result = result+1
                predicted_class.append(result)

    labels = np.array(labels) + 1
    predicted_class = np.array(predicted_class)

    return labels, predicted_class


def run(all_segment_data, templates, total_cycles):
    all_acc = list()
    for subject in all_segment_data.keys():
        labels, predicted_class = tacca_classify(all_segment_data[subject], templates[subject], total_cycles)
        c_mat = confusion_matrix(labels, predicted_class)
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
        all_acc.append(accuracy)
        print('Subject: {subject}, Accuracy: {accuracy} %'.format(subject=subject, accuracy = accuracy*100))


    all_acc = np.array(all_acc)
    print('Overall Accuracy Across Subjects: {all_acc} %, std: {std} %'.format(all_acc=np.mean(all_acc)*100, std=np.std(all_acc)*100))
    return all_acc