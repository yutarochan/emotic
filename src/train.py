'''
EMOTIC: CNN Model Baseline Training
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import emotic_cnn as emotic
from util.data import EMOTICData

# Application Parameters
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
DATA_DIR = ROOT_DIR + '/emotic/'
ANNOT_DIR = ROOT_DIR + '/annotations/Annotations.mat'

# Training Hyperparameters

if __name__ == '__main__':
    print('='*80)
    print('EMOTIC CNN BASELINE MODEL - TRAINING')
    print('='*80 + '\n')

    # Load Dataset
    print('-'*80)
    print('Initialize Dataset and Annotations')
    print('-'*80 + '\n')

    train_data = EMOTICData(DATA_DIR, ANNOT_DIR, 'train')
    valid_data = EMOTICData(DATA_DIR, ANNOT_DIR, 'val')
    test_data  = EMOTICData(DATA_DIR, ANNOT_DIR, 'test')

    # Confirm Dataset Initialization
    print('LOADED: TRAIN SET [' + str(len(train_data)) + ']')
    print('LOADED: VALID SET [' + str(len(valid_data)) + ']')
    print('LOADED: TEST  SET [' + str(len(test_data)) + ']\n')

    # Initialize Image Generator (?)
    # TODO: To include image generator for data augmentation purposes?

    # Initialize Model
    print('-'*80)
    print('Initialize Model')
    print('-'*80 + '\n')

    model = emotic.build_model()
    model.summary()

    # Initialize Training
    print('-'*80)
    print('Training EMOTIC CNN Baseline Model')
    print('-'*80 + '\n')
