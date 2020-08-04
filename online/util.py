from pylsl import StreamInlet, resolve_stream
import scipy.signal as signal
import numpy as np
from sklearn import preprocessing
import math
import stimuli as st
import threading
import pygame
import maze as maze
from sklearn.cross_decomposition import CCA
import serial
from dtw import dtw



def filter(data, Fs):

    if len(data[0,])<1000:
        return None
    data = np.array(data)
    data = data[:7,1000:]
    data = preprocessing.scale(data,axis=1)
    n = 10000
    Fpass1 = 1
    Fpass2 = 42

    n = n | 1
    a = signal.firwin(n, cutoff=[2*Fpass1/Fs, 2*Fpass2/Fs])
    result = []
    for i in range(7):
        conv=signal.convolve(data[i,:], a, 'same')
        result.append(conv)
    data = np.array(result)
    data = data.transpose()
    # t = 1 / Fs:1 / Fs: length(T(:, 1)) / Fs
    return data

def generate_target(data, Fs):
    size = data.shape
    t = np.arange(1 / Fs, (size[0] + 1) / Fs, 1 / Fs)
    target = []
    for i in range(4):
        y= [list(map(lambda x:math.sin(2*math.pi*st.frequencies[i]*x), t )),
            list(map(lambda x:math.cos(2*math.pi*st.frequencies[i]*x), t )),
            list(map(lambda x: math.sin(4 * math.pi * st.frequencies[i] * x), t)),
            list(map(lambda x: math.cos(4 * math.pi * st.frequencies[i] * x), t)),
            list(map(lambda x: math.sin(6 * math.pi * st.frequencies[i] * x), t)),
            list(map(lambda x: math.cos(6 * math.pi * st.frequencies[i] * x), t))]
        y = np.array(y)
        y = np.transpose(y)
        target.append(y)
    return target

def generate_dtw_target(data,Fs):
    size = data.shape
    t = np.arange(1 / Fs, (size[0] + 1) / Fs, 1 / Fs)
    targets = []
    phases = np.array(0, 2*math.pi, 0.5*math.pi)
    for i in range(4):
        phaseTarget = []
        for phase in phases:
            target = list(map(lambda x: math.sin(2 * math.pi * st.frequencies[i] * x + phase), t)),
            phaseTarget.append(target)
        targets.append(phaseTarget)

    return targets

def dtw(data, coefficients, Fs):
    manhattan_distance = lambda x, y: np.abs(x - y)
    X = np.dot(data, coefficients)
    target = generate_dtw_target(data, Fs)
    distance = []
    for fre in target:
        pd = []
        for phase in fre:
            d, cost_matrix, acc_cost_matrix, path = dtw(X, phase, dist=manhattan_distance)
            pd.append(d)
        distance.append(min(pd))


def cca(data, target):
    cca_result = []
    for y in target:
        cca = CCA(n_components=1)
        cca.fit(data,y)

        cca.coef_.shape  # (5,5)

        result = np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)
        result = result[0,1]

        cca_result.append(result)

    index = np.argmax(cca_result)

    return index