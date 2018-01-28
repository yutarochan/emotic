'''
EMOTIC: CNN Model Baseline Training
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import util.data
import emotic_cnn as emotic

# Application Parameters
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
DATA_DIR = ROOT_DIR + '/emotic/'
ANNOT_DIR = ROOT_DIR + '/annotations/Annotations.mat'

# Training Hyperparameters

if __name__ == '__main__':
    # Load Dataset
    print('Initialize Dataset and Annotations')
    train_data = EMOTICData(DATA_DIR, ANNOT_DIR, 'train')
    valid_data = EMOTICData(DATA_DIR, ANNOT_DIR, 'val')
    test_data  = EMOTICData(DATA_DIR, ANNOT_DIR, 'test')

    # Initialize Image Generator (?)
    # TODO: To include image generator for data augmentation purposes?

    # Initialize Model
    print('Initialize Model')
    model = emotic.build_model()
    model.summary()

    # Initialize Training
