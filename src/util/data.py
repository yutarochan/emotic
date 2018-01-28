'''
Emotic: Utility Functions
Auxillary functions to support the emotic program.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt

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
        return len(self.annot[self.MODE][0])

    def __getitem__(self, index):
        # Load Image File
        filename = self.ROOT_DIR + '/' + self.annot[index][1][0] + '/' + self.annot[index][0][0]

        # Extract Image
        bb = self.annot[index][4][0][0][0][0]
        image = np.array(Image.open(open(filename, 'rb')).convert('RGB'))
        body = image[bb[1]:bb[3], bb[0]:bb[2], :]

        # Extract Label
        category = [0] * len(cat_name)
        for x in self.annot[index][4][0][0][1][0][0][0][0]: category[cat_name.index(x[0])] = 1
        vad = [x[0][0] for x in self.annot[index][4][0][0][2][0][0]]

        return ((image, body), (category, vad))

    def get_bb(self, index):
        return self.annot[index][4][0][0][0][0]

if __name__ == '__main__':
    annot = '../../data/annotations/Annotations.mat'
    data = EMOTICData('../../data/emotic/', annot, 'train')

    plt.imshow(data[2][0][0])
    plt.show()

