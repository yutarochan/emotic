'''
Emotic: Utility Functions
Auxillary functions to support the emotic program.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
from PIL import ImageFile
from multiprocessing import Pool

# FIX: Resolve for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# FIX: Supress Warnings
warnings.filterwarnings("ignore")

# Emotion Categories
cat_name = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
            'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
            'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
            'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
            'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

class EMOTICData:
    def __init__(self, root_dir, annotations, mode):
        # Extract Parameters
        self.ROOT_DIR = root_dir
        self.ANOT_DIR = annotations
        self.MODE = mode

        # Load Annotation File
        self.annot = sio.loadmat(self.ANOT_DIR)[self.MODE][0]

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        # Load Image File
        filename = self.ROOT_DIR + '/' + self.annot[index][1][0] + '/' + self.annot[index][0][0]

        # Extract Image
        bb = self.annot[index][4][0][0][0][0]
        image = np.array(Image.open(open(filename, 'rb')).convert('RGB'))
        body = image[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]

        # Extract Label
        category = [0] * len(cat_name)
        for x in self.annot[index][4][0][0][1][0][0][0][0]: category[cat_name.index(x[0])] = 1
        vad = [x[0][0] for x in self.annot[index][4][0][0][2][0][0]]

        return ((image, body), (category, vad))

    def get_(self, index):
        return self[index]

    def get_bb(self, index):
        return self.annot[index][4][0][0][0][0]

    def load_data(self):
        # TODO: Append progress bar using tqdm with multiprocessing - solve pickling issue.
        p = Pool(16)
        data = p.map(self.get_, range(len(self)))
        
        return data

if __name__ == '__main__':
    ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'

    annot = ROOT_DIR + '/annotations/Annotations.mat'
    data = EMOTICData(ROOT_DIR + 'emotic/', annot, 'train')

    train_data = data.load_data()
