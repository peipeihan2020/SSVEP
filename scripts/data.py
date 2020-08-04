import os
import requests
import zipfile

import scipy
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn import preprocessing


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cv2

def save_npz(output, data,labels):
    output_file = os.path.join(cur_dir, output + '.npz')
    if os.path.exists(output_file):
        return
    np.savez(output_file, data=data, labels=labels)
def read_npz(file_path):
    npzfile = np.load(file_path)
    return npzfile['data'], npzfile['labels']

def save_npy(output, data, labels):
    output_file_data = os.path.join(cur_dir, output + '_data.npy')
    if os.path.exists(output_file_data):
        return
    np.save(output_file_data, data)

    output_file_label = os.path.join(cur_dir, output + '_label.npy')
    if os.path.exists(output_file_label):
        return
    np.save(output_file_label, labels)

def read_npy(output):
    output_file_data = os.path.join(cur_dir, output + '_data.npy')
    output_file_label = os.path.join(cur_dir, output + '_label.npy')
    data = np.load(output_file_data, mmap_mode='r')
    labels = np.load(output_file_label, mmap_mode='r')
    return data, labels

def normalize(data, mean, std):
    data = data.astype(np.float32)
    data -=mean
    data /= std
    return data

def get_mean_std(data, eps=1e-8):
    mean = data.mean()
    std = np.std(data)
    return mean, std


def large_data_shuffle():
    train_data,train_labels = read_npy('train')
    test_data, test_labels = read_npy('test')
    save_npz('train',train_data,train_labels)
    save_npy('shuffle/train', train_data,train_labels)
    save_npz('test', test_data, test_labels)
    save_npy('shuffle/test', test_data, test_labels)


def read_images(fold, no):

    data = np.zeros(shape=(no, 97, 97,3), dtype='int32')
    label = np.zeros(shape=(no), dtype='int32')
    for _, dirs, files in os.walk(fold):
        i = 0
        for di in dirs:
            sub = os.path.join(fold, di)
            for subdir, sdir, _ in os.walk(sub):
                for dir in sdir:
                    fre = int(dir)+1
                    subdir = os.path.join(sub, dir)
                    for image_file in os.listdir(subdir):
                        file_path = os.path.join(subdir, image_file)
                        im =cv2.imread(file_path)
                        # im.thumbnail((129,98), Image.ANTIALIAS)
                        # im = cv2.resize( im, (129,98))
                        # plt.imshow(im)
                        # im = misc.imresize(im, (80,80,3))
                        im = im.astype(np.float32)

                        # gray = np.dot(im, [0.2989, 0.5870, 0.1140])
                        # gray = np.expand_dims(gray, axis=-1)
                        data[i,:,:,:] = im
                        label[i]=fre
                        i=i+1
                        if i == no:
                            return data, label


cur_dir = 'img'
train_fold=cur_dir
# test_fold=os.path.join(cur_dir,'')
train_no=5040+2160
# test_no=2160
train_data, train_label= read_images(train_fold, train_no)
mean, std= get_mean_std(train_data)
train_data = normalize(train_data, mean, std)

train_data, train_label= shuffle(train_data, train_label, random_state=0)
# mean, std = get_mean_std(train_data)
# train_data = normalize(train_data, mean, std)


save_npy('train', train_data, train_label)
print('Done with saving training data')
# save_npz('train', train_data, train_label)
# # save_h5py('train', train_data, train_label)
# test_data, test_label= read_images(test_fold, test_no)
# test_data, test_label= shuffle(test_data, test_label, random_state=0)
# test_data = normalize(test_data, mean, std)


# train_data, test_data = white(train_data, test_data)
# print('done with whiten')
# # save_h5py('test', test_data, test_label)
# save_npz('test', test_data, test_label)
# save_npy('train', train_data, train_label)
# save_npy('test', test_data, test_label)
# print('Done with saving testing data')
# large_data_shuffle()