import class_data_12 as tldata
import cca
import itcca
import ccca
from numpy import asarray
from numpy import savetxt
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def save_csv(data, path):
    savetxt(path, data, delimiter=',')

def plot_results():
    times = np.arange(1*0.2, 6*0.2, 0.2)[:5]
    results = dict()
    results_single = dict()
    path = '../results/'
    for f in sorted(os.listdir(path)):
        name = os.path.splitext(f)[0].split('_')
        if len(name) ==2:
            key = name[0]
            results.setdefault(key, [])
        else:
            key = name[0]+name[1]
            results_single.setdefault(key, [])


        file_path = os.path.join(path, f)
        with open(file_path, newline='') as csvfile:
            data =np.array(list(csv.reader(csvfile)))
            data = np.squeeze(data).astype(np.float)
            if key in results.keys():
                results[key].append(np.mean(data)*100)
            else:
                results_single[key].append(np.mean(data)*100)

    # results['C_CNN'] = [35.22486784309149, 59.39899002015591, 80.84920611977577, 75.36111125349998,90.5972221493721 ]
    results['C_CNN'] = [32.22222223877907, 56.76767736673355, 75.19841313362122, 71.99074119329453, 74.30555641651154]
    plot_data(results, times, 'Window Length(s)', 'Average Accuracy(%)', [0,1, 0, 100])

    shifts = np.arange(0.1, 0.9, 0.1)
    harmonics = range(1, 5, 1)
    for key in results_single.keys():
        if 'shift' in key:
            plot_one(results_single[key], shifts, 'Overlap Window Length(s)', 'Average Accuracy(%)', [0, 1, 0, 100])
        elif 'harmonics' in key:
            plot_one(results_single[key], harmonics, 'Number of Harmonics', 'Average Accuracy(%)', [0, 5, 0, 100])

def plot_one(result, domains, xlabel, ylabel, range):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(range)
    fig, ax = plt.subplots()
    ax.plot(domains, result)
    ax.plot(domains, result, 'D')
    ax.legend()
    plt.show()


def plot_data(results, times, xlabel, ylabel, range):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(range)
    fig, ax = plt.subplots()
    for key in results.keys():
        label = key
        if key == 'ccca':
            label = 'Extended CCA Method'
        elif key == 'itcca':
            label = 'IT-CCA'
        elif key == 'C_CNN':
            label = 'Complex Spectrum CNN'
        ax.plot(times, results[key], label=label.upper())
        ax.plot(times, results[key], 'D')
    ax.legend()
    plt.show()

def run_cca_by_shift():
    win_len = 1
    shifts = np.arange(0.1, 0.9,0.1)
    i=1
    for shift_len in shifts:
        data, filtered_data = tldata.get_data(win_len, shift_len)
        targets = tldata.get_targets(win_len,2)
        accuracy = cca.run(data, targets)
        save_csv(accuracy, '../results/cca_shift_{i}.csv'.format(i = i))
        i +=1

def run_cca_by_harmonics():
    win_len = 1
    shift_len = win_len

    for i in range(1,5,1):
        data, filtered_data = tldata.get_data(win_len, shift_len)
        targets = tldata.get_targets(win_len, i)
        accuracy = cca.run(data, targets)
        save_csv(accuracy, '../results/cca_harmonics_{i}.csv'.format(i = i))
        i +=1

def run_cca_by_filters():
    win_len = 1
    shift_len = win_len
    data, filtered_data = tldata.get_data(win_len, shift_len, 'elliptic')
    targets = tldata.get_targets(win_len, 2)
    accuracy = cca.run(data, targets)


def main():
    for i in range(1,6):
        win_len = i*0.2
        shift_len= win_len
        data, filtered_data =  tldata.get_data(win_len, shift_len)
        targets = tldata.get_targets(win_len, 2)
        templates = tldata.get_templates(filtered_data, win_len, shift_len)
        # accuracy = cca.run(data, targets)
        # save_csv(accuracy, '../results/cca_{i}.csv'.format(i=i))
        total_cycles = 14
        accuracy = itcca.run(data, templates, total_cycles)
        save_csv(accuracy, '../results/itcca_{i}.csv'.format(i=i))

        accuracy = ccca.run(data, templates, targets, total_cycles)
        save_csv(accuracy, '../results/ccca_{i}.csv'.format(i=i))

if __name__ == '__main__':
    # main()
    plot_results()
    # run_cca_by_shift()
    # run_cca_by_harmonics()
    # plot_results()
    # run_cca_by_filters()