'''
Emotic: Plot Functions
Auxillary functions to visualize dataset and annotations.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from data import EMOTICData

def plot_im(image):
    plt.imshow(image[0][0])
    plt.show()

def plot_body(image):
    plt.imshow(image[0][1])
    plt.show()

def plot_antbod(image, bb):
    fig, ax = plt.subplots(1)
    ax.imshow(image[0][0])
    rect = pat.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[0], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    annot = '../../data/annotations/Annotations.mat'
    data = EMOTICData('../../data/emotic/', annot, 'train')

    plot_im(data[0])
    plot_body(data[0])
    plot_antbod(data[0], data.get_bb(0))

