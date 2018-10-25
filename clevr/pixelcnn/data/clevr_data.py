import numpy as np
from urllib import request
import gzip
import pickle

import os
import sys
import tarfile
from six.moves import urllib
import itertools
from glob import glob 
from scipy import misc
import random

class DataLoader(object):
    """ an object that generates batches of clevr data for training """

    def __init__(self, batch_size, data_str, full, dataset_name, cors, rng=None, shuffle=False, return_labels=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_str = data_str
        self.dataset_name = dataset_name
        self.cors = cors
        self.input_fname_pattern = "*.png"
        self.full = full.split('-')
        assert len(self.full) == 2
        self.color_set = self.full[0].split('.')#['red', 'blue', 'brown', 'green']
        self.shape_set = self.full[1].split('.')#['corn', 'cylinder', 'sphere', 'torus']
        # print (self.color_set)
        # print (self.shape_set)

        self.universal_set_color = list(itertools.product(self.color_set, self.color_set))
        self.universal_set_shape = list(itertools.product(self.shape_set, self.shape_set))
        self.universal_set = list(itertools.product(self.universal_set_color, self.universal_set_shape))
        self.train_set = []
        self.delete_set = []
        self.excp_data = self.dataset_name.split('-')
        self.excp_data = [i.split('.') for i in self.excp_data]
        self.dis_data = []
        for x in self.excp_data:
            for i in x:
                self.dis_data.append(i.split('_'))
        self.excp_data = self.dis_data
        assert len(self.excp_data) % 2 == 0, "wrong exceptions, odd not allowed"
        for i in range(len(self.excp_data)):
            assert self.excp_data[i][0] in self.color_set and self.excp_data[i][1] in self.shape_set

        # self.excp_data = self.dataset_name.split('.')
        # self.excp_data = [i.split('_') for i in self.excp_data]
        if cors == 'color':
            for i in range(len(self.excp_data)):
                if i%2 != 0: continue
                self.delete_set.append('%s.%s'%(self.excp_data[i][0], self.excp_data[i+1][0]))
        elif cors == 'shape':
            for i in range(len(self.excp_data)):
                if i%2 != 0: continue
                self.delete_set.append('%s.%s'%(self.excp_data[i][1], self.excp_data[i+1][1]))
        else:
            assert False, "please choose color or shape as CORS"
        # print (self.delete_set)

        for i in self.universal_set:
            if cors == 'color':
                if '%s.%s'%(i[0][0], i[0][1]) in self.delete_set:
                    continue
            else:
                if '%s.%s'%(i[1][0], i[1][1]) in self.delete_set:
                    continue
            self.train_set.append('%s_%s.%s_%s'%(i[0][0], i[1][0], i[0][1], i[1][1]))
        # self.train_set.append(self.dataset_name)
        for i in range(len(self.excp_data)):
            if i%2 != 0: continue
            self.train_set.append('%s_%s.%s_%s'%(self.excp_data[i][0], self.excp_data[i][1], self.excp_data[i+1][0], self.excp_data[i+1][1]))
        print(len(self.train_set))
        self.dataset_config = self.train_set # dataset_name.split('-')

        self.data = []
        for d_str in self.dataset_config:
            print(os.path.join(self.data_str, d_str, 'images/', self.input_fname_pattern))
            self.data.extend(glob(os.path.join(self.data_str, d_str, 'images/', self.input_fname_pattern)))
            kk = glob(os.path.join(self.data_str, d_str, 'images/', self.input_fname_pattern))
            assert len(kk) == 400

        self.images = np.array([misc.imresize(misc.imread(a), [32,32,4]) for a in self.data])
        self.images = self.images[:,:,:,:3]
        # self.images = np.array([misc.imresize(a, [32,32,3]) for a in self.images])
        print (self.images.shape)
        # for i in range(10):
        #     misc.imsave('test/%d.png'%i, self.images[i])
        # abc
        # t = self.images[0].copy()
        # t = misc.imresize(t, [32,32,3])
        # misc.imsave('test/100.png', t)
        # abc
        # self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # load CIFAR-10 training data to RAM
        # self.data, self.labels = load(data_dir, subset=subset)
        # self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
        self.data = self.images # np.reshape(self.data, (-1, 128, 128, 3))
        self.labels = np.ones((len(self.images)))
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
