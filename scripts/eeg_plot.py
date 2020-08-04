import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from pathlib import Path

from matplotlib.collections import LineCollection

import os
import sys

import numpy as np
import scipy.io as sio
import preprocess as su




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




def plot(data, target,index, subject):
    fig = plt.figure(figsize=(1,1))
    data = data.T
    n_samples, n_rows = data.shape

    t = 10 * np.arange(n_samples) / n_samples

    # Plot the EEG
    ticklocs = []
    ax2 = fig.gca()
    ax2.set_xlim(0, 10)
    # ax2.set_xticks(np.arange(10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    ax2.set_ylim(y0, y1)

    segs = []
    for i in range(n_rows):
        segs.append(np.column_stack((t, data[:, i])))
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax2.add_collection(lines)
    ax2.axis('off')
    ax2.margins(0)
    img_path = 'img/'+str(subject)+'/'+str(target)+'/'
    os.makedirs(img_path, exist_ok=True)
    # Path(img_path).mkdir(parents=True, exist_ok=True)
    img_path =img_path +str(index)+'.png'

    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

def save_data_to_img():
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
    index = 1
    for subject in all_segment_data.keys():
        segmented_data = all_segment_data[subject]
        for target in range(0, segmented_data.shape[0]):
            for trial in range(0, segmented_data.shape[2]):
                for segment in range(0, segmented_data.shape[3]):
                    plot(segmented_data[target, :, trial, segment, :], target, index, subject)
                    index += 1

save_data_to_img()