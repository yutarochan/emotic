'''
MATLAB to CSV File Converter
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import time
import torch.nn as nn
import numpy as np
import scipy.io as sio

# Emotion Categories                                                                                                                                                                                                                                                                          
cat_name = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
            'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
            'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
            'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
            'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

class EMOTICData(nn.Module):
    def __init__(self, root_dir, annotations, mode):
        # Extract Parameters                                                                                                                                                                                                  
        self.ROOT_DIR = root_dir
        self.ANOT_DIR = annotations
        self.MODE = mode

        # Load Annotation File
        start = time.time()
        self.annot = sio.loadmat(self.ANOT_DIR)[self.MODE][0]
        end = time.time()

        # Print Statement                                                                                                                                                                                  
        print('LOADED', self.MODE, '\t[', len(self.annot),']\t', (end - start), ' sec.')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        # Load Image File                                                                                                                                                                                                              
        filename = self.ROOT_DIR + '/' + self.annot[index][1][0] + '/' + self.annot[index][0][0]

        # Extract Image                                                                                                                                                                                                              
        bb = self.annot[index][4][0][0][0][0]

        # Extract Label                                                                                              
        category = np.array([0] * len(cat_name), dtype=np.float64)
        if self.MODE == 'train':
            for x in self.annot[index][4][0][0][1][0][0][0][0]: category[cat_name.index(x[0])] = 1
            vad = np.array([x[0][0] for x in self.annot[index][4][0][0][2][0][0]])
        else:
            for x in self.annot[index][4][0][0][2][0]: category[cat_name.index(x[0])] = 1
            vad = np.array([x[0][0] for x in self.annot[index][4][0][0][4][0][0]])
            
        # Perform NaN Checks
        if len(vad) != 3: return None

        return (filename, bb.tolist(), category.tolist(), vad.tolist())

# CSV File Writer
def csv_writer(data_loader, csv_writer, remove_nan=True):
    print(len(data_loader))
    for i in range(len(data_loader)):
        data = data_loader[i]
        if data is None: print("Dat Corrupted!")
        if remove_nan and data is None: continue
        output = data[0] + ','
        output += ','.join([str(i) for i in data[1]]) + ','
        output += ','.join([str(i) for i in data[2]]) + ','
        output += ','.join([str(i) for i in data[3]]) + '\n'
        csv_writer.write(output)

# Define Application Parameters
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
ANOT_DIR = ROOT_DIR + '/annotations/Annotations.mat'
REMOVE_NAN = True

# Load Dataset
train_data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'train')
valid_data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'val')
test_data  = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'test')


train_out = open(ROOT_DIR + 'emotic/train_annot.csv', 'w')
csv_writer(train_data, train_out, REMOVE_NAN)
train_out.close()

valid_out = open(ROOT_DIR + 'emotic/valid_annot.csv', 'w')
csv_writer(valid_data, valid_out, REMOVE_NAN)
valid_out.close()

test_out  = open(ROOT_DIR + 'emotic/test_annot.csv', 'w')
csv_writer(test_data, test_out, REMOVE_NAN)
test_out.close()

print('Done')
