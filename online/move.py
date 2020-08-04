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
import util as ul
import realtime_data as rldata
from scripts import preprocess as pre
import ccca

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
Fs = inlet.info().nominal_srate()
runing=True
trainning = False
move_delay = 1000
save_training = False
label = -1
import os
curDir = '../data/subjects'
training_period = 4000
fileName = None
saved = False
sample_rate = 500


def first_buffer():
    chunk = read_data()
    chunk=np.array(chunk).transpose()
    if len(chunk) == 0:
        return False, None
    bufferdata = []
    result = False
    if(any(chunk[7,:])):
        result = True
        arr = np.array(chunk)
        index = np.where(arr>0)
        if index is not None:
            if len(index[1])>0:
                bufferdata = chunk[:,index[1][0]:]
    return result, bufferdata



def read_data():
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    chunk, timestamps = inlet.pull_chunk()
    return chunk

def save_npz(output, data,labels):
    output_file = os.path.join(curDir, output + '.npz')
    if os.path.exists(output_file):
        return
    np.savez(output_file, data=data, labels=labels)

def thread_save():
    global saved
    templates = rldata.get_templates()
    references = rldata.get_targets(1,1)
    while runing:
        if not trainning:
            moveon, bufferData = first_buffer()
            if not moveon:
                continue

        else:
            read_data()
            if not save_training:
                continue
            bufferData = None

        current_time = pygame.time.get_ticks()
        # how long to show or hide

        # time of next change
        if not trainning:
            change_time = current_time + move_delay
            samples = sample_rate * (move_delay/1000)
        else:
            change_time = current_time + training_period
            samples = sample_rate * (training_period / 1000)


        while current_time < change_time or bufferData.shape[1] < samples:
            chunk= read_data()
            if len(chunk) > 0:
                chunk = np.array(chunk).transpose()
                if bufferData is None:
                    bufferData = chunk
                bufferData = np.concatenate((bufferData, chunk),axis=1)
            current_time = pygame.time.get_ticks()

        # write_to_com(0)
        if not trainning:
            move(bufferData, templates, references)
        elif save_training:
            save_npz(fileName, bufferData)
            saved = True
        # write_to_com(1)



def move(bufferData,template, reference):
    try:
        data = rldata.get_data(bufferData)

        index = ccca.run(data, template, reference)
        player = maze.player
        if player is None:
            raise Exception('Fail find player')
        maze.move_right()
        # if index == 0:
        #     maze.move_up()
        # elif index == 1:
        #     maze.move_bottom()
        # elif index == 2:
        #     maze.move_left()
        # else:
        #     maze.move_right()
    except Exception as error:
        print(error)



def save_data(train=False):
    global runing
    global trainning
    trainning = train
    runing = True
    x = threading.Thread(target=thread_save, daemon=True)
    x.start()
    return x

def stop(x):
    global runing
    runing= False
    x.join()


def write_to_com(data):
    with serial.Serial() as ser:
        ser.baudrate = 57600
        ser.port = 'COM1'
        ser.open()
        ser.write(data)