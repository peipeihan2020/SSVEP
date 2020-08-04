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

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
Fs = inlet.info().nominal_srate()
runing=True


def first_buffer():
    chunk = read_data()
    chunk=np.array(chunk).transpose()
    if len(chunk) == 0:
        return False, None
    bufferdata = []
    result = False
    if(any(chunk[11,:])):
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

def thread_save():
    while runing:
        moveon, bufferData = first_buffer()
        if not moveon:
            continue
        current_time = pygame.time.get_ticks()

        # how long to show or hide
        move_delay = 7000

        # time of next change
        change_time = current_time + move_delay
        while current_time < change_time:
            chunk= read_data()
            if len(chunk) > 0:
                chunk = np.array(chunk).transpose()
                bufferData = np.concatenate((bufferData, chunk),axis=1)
            current_time = pygame.time.get_ticks()


        try:
            data = ul.filter(bufferData,Fs)
            target = ul.generate_target(data, Fs)
            index = ul.cca(data, target)
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


def save_data():
    global runing
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