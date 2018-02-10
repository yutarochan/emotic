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
from util.data import EMOTICData
from emotic import EmoticCNN, DiscreteLoss

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    
    # TODO: Improve Loss Function Metric Tracking
    # train_loss = 0
    # correct = 0
    # total = 0
    
    # TODO: Add a better UI for tracking the training progress of a model.
    for batch_idx, data in enumerate(data_loader):
        print('BATCH: ' + str(batch_idx))
        # TODO: Fix underlying data loading to remove any NaN samples.

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

        # Compute Loss Weight
        # d_weight = emotic.disc_weight(disc_pred, disc.float())

        # Initialize Weighted Loss Functions
        # disc_loss = nn.MSELoss(size_average=True)
        disc_loss = emotic.DiscreteLoss()

        # Compute Loss & Backprop
        d_loss = disc_loss(disc_pred, disc.float())
        d_loss.backward()

        optimizer.step()

        print(d_loss.data[0])
        # train_loss += d_loss.data[0]
    # print('LOSS: ' + str(train_loss))

if __name__ == '__main__':
    print('='*80)
    print('EMOTIC CNN BASELINE MODEL - TRAINING')
    print('='*80 + '\n')

    # Load Dataset
    print('-'*80)
    print('Initialize Dataset and Annotations')
    print('-'*80)

    # Data Transformation and Normalization
    # TODO: Check performance difference between Scale vs Resizing of Image
    # TODO: Check to see if we have properly normalized the image.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([transforms.Scale(params.IM_DIM), transforms.ToTensor()])

    # Load Dataset Generator Objects
    train_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR, 'train', transform=transform)
    valid_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR, 'val', transform=transform)
    test_data  = EMOTICData(params.DATA_DIR, params.ANNOT_DIR, 'test', transform=transform)
    
    # Initialize Dataset Loader Objects
    train_loader = DataLoader(train_data, batch_size=params.TRAIN_BATCH_SIZE, shuffle=params.TRAIN_DATA_SHUFFLE, num_workers=params.NUM_WORKERS)
    valid_lodaer = DataLoader(valid_data, batch_size=params.VALID_BATCH_SIZE, shuffle=params.VALID_DATA_SHUFFLE, num_workers=params.NUM_WORKERS)
    test_loader  = DataLoader(test_data,  shuffle=False, num_workers=params.NUM_WORKERS)

    # Initialize Model
    print('-'*80)
    print('Initialize Model')
    print('-'*80)

    model = emotic.EmoticCNN()
    
    # Display Model Module Summary Information
    # emotic.model_summary(model)

    # IF USE_CUDA - Set model parameters compatible against CUDA GPU
    # TODO: Figure method for multiple GPU use
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
    
    # Define Cost Functions and Optimization Criterion
    # disc_loss = emotic.DiscreteLoss()
    # cont_loss = None

    # Initialize Optimizer
    optimizer = optim.SGD(model.parameters(), lr=params.TRAIN_LR, momentum=0.9)
    
    # Perfom Training Iterations
    for epoch in range(params.START_EPOCH, params.TRAIN_EPOCH+1):
        print('EPOCH ',epoch,'/',params.TRAIN_EPOCH)
        train_epoch(epoch, None, model, train_loader, optimizer)
        
        break
