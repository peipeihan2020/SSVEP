import numpy as np
from sklearn.metrics import confusion_matrix
from scripts import ssvep_utils as su


def cca_classify(segmented_data, target_signals):
    '''
        Returns predicted class and lables.

        Args:
            segmented_data (numpy.ndarray): segemented EEG signals.
            target_signals (numpy.ndarray): target frequency list.

        Returns:
            (numpy.ndarray): labels.
            (numpy.ndarray): predicted labels.
    '''
    predicted_class = []
    labels = []
    for target in range(0, segmented_data.shape[0]):
        for trial in range(0, segmented_data.shape[2]):
            for segment in range(0, segmented_data.shape[3]):
                labels.append(target)
                result, *_ = su.find_correlation(1, segmented_data[target, :, trial, segment, :],
                                          target_signals)
                predicted_class.append(result + 1)
    labels = np.array(labels) + 1
    predicted_class = np.array(predicted_class)

    return labels, predicted_class


def run(all_segment_data, target_signals):
    '''
        Returns average accuracy.

        Args:
            all_segment_data (numpy.ndarray): segemented EEG signals.
            target_signals (numpy.ndarray): target frequency list.

        Returns:
            (float): accuray.
    '''
    all_acc = list()
    for subject in all_segment_data.keys():
        labels, predicted_class = cca_classify(all_segment_data[subject], target_signals)
        c_mat = confusion_matrix(labels, predicted_class)
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
        all_acc.append(accuracy)
        print('Subject: {subject}, Accuracy: {accuracy} %'.format(subject=subject, accuracy = accuracy*100))


    all_acc = np.array(all_acc)
    print('Overall Accuracy Across Subjects: {all_acc} %, std: {std} %'.format(all_acc=np.mean(all_acc)*100, std=np.std(all_acc)*100))
    return all_acc