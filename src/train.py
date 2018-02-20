'''
EMOTIC: CNN Model Baseline Training
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import time
import params
import emotic
import numpy as np
from tqdm import tqdm
import torchnet as tnt
from util.meter import AverageMeter
from util.data_csv import EMOTICData
from torchnet.logger import MeterLogger
from emotic import EmoticCNN, DiscreteLoss

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize Auxillary Functions and Utilities
if params.VISTORCH_LOG: 
    mlog = MeterLogger(server='localhost', port=8097, title="EMOTIC CNN Baseline")

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    
    # TODO: Setup better metric evaluation class to handle data logging and visualization to visdom.
    train_dloss = 0
    train_closs = 0
    # correct = 0
    # total = 0

    # losses = AverageMeter()

    for i, data in enumerate(tqdm(data_loader)):
        # Initialize Data Variables
        image = Variable(data[0], requires_grad=True)
        body  = Variable(data[1], requires_grad=True)
        disc  = Variable(data[2], requires_grad=False)
        cont  = Variable(data[3], requires_grad=False)
        
        # Utilize CUDA
        if params.USE_CUDA:
            body, image, disc, cont = body.cuda(), image.cuda(), disc.cuda(), cont.cuda()
        
        # Generate Predictions
        optimizer.zero_grad()
        disc_pred, cont_pred = model(body, image)

        # Initialize Weighted Loss Functions
        disc_loss = emotic.DiscreteLoss()
        cont_loss = emotic.ContinuousLoss()

        # Compute Loss & Backprop
        # d_loss = disc_loss(pr
        '''
        d_loss = disc_loss(disc_pred, disc.float())
        d_loss.backward()

        optimizer.step()

        c_loss = cont_loss(cont_pred, cont.float())
        c_loss.backward()
        optimizer.step()
        '''
        
        # Record Accuracy and Loss Data
        train_dloss += d_loss.data[0]
        train_closs += c_loss.data[0]

        print(d_loss.data)

        # Update Log Dashboard Data Point
        if params.VISTORCH_LOG:
            mlog.update_loss(d_loss.data, meter='Discrete Loss')
        
    # Update Plot
    print('D_LOSS: ' + str(train_dloss) + '\tC_LOSS: ' + str(train_closs))

    if params.VISTORCH_LOG:
        mlog.print_meter(mode="Train", iepoch=epoch)
        mlog.reset_meter(mode="Train", iepoch=epoch)

    # TODO: Implement Intermediate Model Persistencee

if __name__ == '__main__':
    print('='*80)
    print('EMOTIC CNN BASELINE MODEL - TRAINING')
    print('='*80 + '\n')

    # Load Dataset
    print('-'*80)
    print('Initialize Dataset and Annotations')
    print('-'*80)

    # Data Transformation and Normalization
    # TODO: Use the normalization constants from the emotic_tf codebase!
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Scale(params.IM_DIM), transforms.ToTensor()])

    # Load Dataset Generator Objects
    train_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_TRAIN, transform=transform)
    valid_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_VALID, transform=transform)
    test_data  = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_TEST,  transform=transform)
    
    # Initialize Dataset Loader Objects
    train_loader = DataLoader(train_data, batch_size=params.TRAIN_BATCH_SIZE, shuffle=params.TRAIN_DATA_SHUFFLE, num_workers=params.NUM_WORKERS)
    valid_loader = DataLoader(valid_data, batch_size=params.VALID_BATCH_SIZE, shuffle=params.VALID_DATA_SHUFFLE, num_workers=params.NUM_WORKERS)
    test_loader  = DataLoader(test_data,  shuffle=False, num_workers=params.NUM_WORKERS)

    # Initialize Model
    print('-'*80)
    print('Initialize Model')
    print('-'*80)

    model = emotic.EmoticCNN()
    
    # Display Model Module Summary Information
    # emotic.model_summary(model)

    # IF USE_CUDA - Set model parameters compatible against CUDA GPU
    if params.USE_CUDA:
        print('>> CUDA USE ENABLED!')
        print('>> GPU DEVICES AVAILABLE COUNT: ' + str(torch.cuda.device_count()))
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Initialize Training
    print('\n'+'-'*80)
    print('Training EMOTIC CNN Baseline Model')
    print('-'*80)
    
    # Initialize Optimizer
    # TODO: Parameterize different methods for optimizers
    optimizer = optim.SGD(model.parameters(), lr=params.TRAIN_LR, momentum=0.9)
    
    # Perfom Training Iterations
    for epoch in range(params.START_EPOCH, params.TRAIN_EPOCH+1):
        print('EPOCH '+str(epoch)+'/'+str(params.TRAIN_EPOCH))
        train_epoch(epoch, None, model, train_loader, optimizer)
