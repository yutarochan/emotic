'''
EMOTIC CNN Baseline: Parameter Configuration
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function

''' APPLICATION PARAMETERS '''
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
DATA_DIR = ROOT_DIR + '/emotic/'
ANNOT_DIR = ROOT_DIR + '/annotations/Annotations.mat'

USE_CUDA = True
NUM_WORKERS = 8

''' APPLICATION CONSTANTS '''
NDIM_DISC = 26
NDIM_CONT = 3

IM_DIM = (256, 256)

''' TRAINING PARAMETERS '''
BN_EPS = 0.001
STD_VAR_INIT = 1e-2
TRAIN_LR = 0.001

TRAIN_BATCH_SIZE = 52
TRAIN_DATA_SHUFFLE = True

VALID_BATCH_SIZE = 52
VALID_DATA_SHUFFLE = True

START_EPOCH = 1
TRAIN_EPOCH = 100

# Loss Parameters
W_CONT = 1.0
W_DISC = 1.0/6.0
LOSS_CONT_MARGIN = 0.1
LDISC_C = 1.2

''' TESTING PARAMETERS '''
TEST_BATCH_SIZE = 100
