'''
EMOTIC Dataset Utility
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import time
import torch
import warnings
import numpy as np
from PIL import Image
import scipy.io as sio
from PIL import ImageFile
from scipy.misc import imresize
from multiprocessing import Pool
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

# FIX: Truncated Image Error
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: Implement multithreaded GPU import data generator.
# https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
# https://www.kaggle.com/danielhavir/pytorch-dataloader

# TODO: Implement thread safe method for concurrency handling and data IO pipelining to improve data load procedure to mask behind GPU.

# Emotion Categories                                                                                                                                                                                                                        
cat_name = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
            'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
            'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
            'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
            'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

class EMOTICData(Dataset):
    def __init__(self, root_dir, annotations, mode, transform):
        # Extract Parameters
        self.ROOT_DIR = root_dir
        self.ANOT_DIR = annotations
        self.MODE = mode
        self.transform = transform
        
        # Load Annotation File
        start = time.time() 
        self.annot = sio.loadmat(self.ANOT_DIR)[self.MODE][0]
        end = time.time()

        # Print Statement
        print('LOADED ', self.MODE, ' [', len(self.annot), '] ', (end - start), ' sec.')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        # Load Image File
        filename = self.ROOT_DIR + '/' + self.annot[index][1][0] + '/' + self.annot[index][0][0]

        # Extract Image
        bb = self.annot[index][4][0][0][0][0]
        image = Image.open(open(filename, 'rb')).convert('RGB')
        if image == None: print('NONE DETECTED! ' + str(filename))
        body = image.crop((int(bb[0]), int(bb[1]), int(bb[3]), int(bb[2])))

        # Extract Label
        category = np.array([0] * len(cat_name))
        for x in self.annot[index][4][0][0][1][0][0][0][0]: category[cat_name.index(x[0])] = 1
        vad = np.array([x[0][0] for x in self.annot[index][4][0][0][2][0][0]])

        # Perform Data Transformations
        if self.transform:
            image = self.transform(image)
            body = self.transform(body)
        
        return ([image, body], [category, vad])

if __name__ == '__main__':
    ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
    ANOT_DIR = ROOT_DIR + '/annotations/Annotations.mat'

    # Data Transformation and Normalization
    # TODO: Define a better normalization scheme...
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Data Transformation and Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Load Training Data
    start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'train', transform=transform)
    end = time.time()
    print('Train - Total Time Elapsed: ' + str(end - start) + ' sec.')

    sample = data[0]

    '''
    # Load Validation Data
    start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'val')
    end = time.time()
    print('Validation - Total Time Elapsed: ' + str(end - start) + ' sec.')

    # Load Testing Data
    start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'test')
    end = time.time()
    print('Validation - Total Time Elapsed: ' + str(end - start) + ' sec.')
    '''
