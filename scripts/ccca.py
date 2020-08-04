import os
import sys
from dtw import dtw, accelerated_dtw
import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
import preprocess as pre
import math

from scripts import ssvep_utils as su

def get_cor_template(X, ref, xweights, yweights):
    corr = []
    X = X.T
    Y =  np.squeeze(ref).T
    X_r = np.dot(X, xweights)
    Y = np.dot(Y, yweights)
    return np.corrcoef(X_r[:,0], Y[:,0])[0, 1]


def ccca_classify(segmented_data,templates, reference_signals, total_cycles):
    predicted_class = []
    labels = []

    n_components =1
    cca = CCA(n_components)
    # templates = get_template_signals(segmented_data, total_cycles)
    # templates = np.array(templates)
    for t in range(0, segmented_data.shape[0]):
        for trial in range(total_cycles, segmented_data.shape[2]):
            for segment in range(0, segmented_data.shape[3]):
                labels.append(t)
                refe = templates[:, :, :, segment, :]
                corrs = []
                for target in range(0, segmented_data.shape[0]):
                    results = []
                    X = segmented_data[t, :, trial, segment, :]
                    cor,_, xweights, yweights = su.find_correlation_for_one_pair(cca,1, segmented_data[t, :, trial, segment, :], refe[target, :, :])
                    cor_ref, _, xweights_ref, yweights_ref  = su.find_correlation_for_one_pair(cca, 1, segmented_data[t, :, trial, segment, :],
                                                     reference_signals[target, :, :])
                    cor_ref_tem, _, xweights_ref_tem, yweights_ref_tem = su.find_correlation_for_one_pair(cca, 1,
                                                                                            np.squeeze(refe[target, :, :]),
                                                                                 reference_signals[target, :, :])
                    corr_t = get_cor_template(X, refe[target, :, :], xweights_ref, xweights_ref)
                    corr_ref_temp = get_cor_template(X, refe[target, :, :], xweights_ref_tem, xweights_ref_tem )
                    corr_temp = get_cor_template(np.squeeze(refe[target, :, :]), refe[target, :, :], xweights, yweights)

                    results = [cor, cor_ref, corr_t, corr_ref_temp, corr_temp]
                    corsum =map(lambda x:np.sign(x)* x**2, results)
                    corsum = list(corsum)
                    corsum = np.sum(corsum)
                    corrs.append(np.sum(corsum))
                pre= np.argmax(corrs)+1
                predicted_class.append(pre)

    labels = np.array(labels) + 1
    predicted_class = np.array(predicted_class)

    return labels, predicted_class



def run(all_segment_data,reference_templates,reference_signals, total_cycles ):
    all_acc = list()
    for subject in all_segment_data.keys():
        labels, predicted_class = ccca_classify(all_segment_data[subject], reference_templates[subject], reference_signals, total_cycles)
        c_mat = confusion_matrix(labels, predicted_class)
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
        all_acc.append(accuracy)
        print('Subject: {subject}, Accuracy: {accuracy} %'.format(subject=subject, accuracy = accuracy*100))


    all_acc = np.array(all_acc)
    print('Overall Accuracy Across Subjects: {all_acc} %, std: {std} %'.format(all_acc=np.mean(all_acc)*100, std=np.std(all_acc)*100))
    return all_acc