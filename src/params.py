'''
EMOTIC CNN Baseline: Parameter Configuration
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function

''' APPLICATION PARAMETERS '''
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
DATA_DIR = ROOT_DIR + '/emotic/'

ANNOT_DIR_TRAIN = ROOT_DIR + '/emotic/train_annot.csv'
ANNOT_DIR_VALID = ROOT_DIR + '/emotic/valid_annot.csv'
ANNOT_DIR_TEST  = ROOT_DIR + '/emotic/test_annot.csv'

USE_CUDA = True
NUM_WORKERS = 8

''' APPLICATION CONSTANTS '''
NDIM_DISC = 26
NDIM_CONT = 3

IM_DIM = (256, 256)

''' TRAINING PARAMETERS '''
BN_EPS = 0.001
STD_VAR_INIT = 1e-2
TRAIN_LR = 0.1

TRAIN_BATCH_SIZE = 4
TRAIN_DATA_SHUFFLE = False

VALID_BATCH_SIZE = 52
VALID_DATA_SHUFFLE = True

START_EPOCH = 1
TRAIN_EPOCH = 1

# Loss Parameters
W_CONT = 1.0
W_DISC = 1.0/6.0
LOSS_CONT_MARGIN = 0.1
LDISC_C = 1.2

''' TESTING PARAMETERS '''
TEST_BATCH_SIZE = 100
