'''
EMOTIC Dataset Utility [CSV Version]
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import time
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from PIL import ImageFile
from scipy.misc import imresize
from multiprocessing import Pool
from torch.autograd import Variable
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
    def __init__(self, root_dir, annotations, transform):
        # Extract Parameters
        self.ROOT_DIR = root_dir
        self.ANOT_DIR = annotations
        self.transform = transform
        
        # Load Annotation File
        start = time.time()
        self.annot = pd.read_csv(self.ANOT_DIR, skiprows=0).dropna()
        end = time.time()

        # Print Statement
        print('LOADED', self.ANOT_DIR, '\t[', len(self.annot),']\t', (end - start), ' sec.')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        # Load Image File
        filename = self.annot.iloc[index, 0]
        
        # Extract Image
        bb = self.annot.iloc[index, [1,2,3,4]].tolist()
        image = Image.open(open(filename, 'rb')).convert('RGB')
        body = image.crop((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))

        # Extract Label
        category = np.array(self.annot.iloc[index, range(5, 31)].astype(int).tolist())
        vad = np.array(self.annot.iloc[index, range(31, 34)].astype(int).tolist()) / 10.0

        # Perform Data Transformations
        if self.transform:
            image = self.transform(image)
            body = self.transform(body)
        
        return (image, body, category, vad)

if __name__ == '__main__':
    ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
    ANOT_DIR = ROOT_DIR + '/emotic/train_annot.csv'

    # Data Transformation and Normalization
    # TODO: Define a better normalization scheme...
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Data Transformation and Normalization
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Load Training Data
    # start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, transform=transform)
    print(data[325])
    # end = time.time()
    # print('Train - Total Time Elapsed: ' + str(end - start) + ' sec.')

    # sample = data[0]
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

    # Initialize Data Loader
    # data_loader = torch.utils.data.DataLoader(data)

    # Dataset Batch Loading
    # for batch_idx, sample in enumerate(data_loader):
        # image = Variable(sample[0])
        # print(sample[3])

        # print(len(sample))
        # print(sample[batch_idx][3])

    # print(data[263]) # Test Load
