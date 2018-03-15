'''
EMOTIC: CNN Model Baseline Training
Author: Yuya Jeremy Ong (yjo5006@psu.edu)

TODO: Rewrite every fucking piece of shit...
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
import copy
import params
import emotic
import numpy as np
import util.loss_log
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

# Setup Best Model Tracker
BEST_MODEL = copy.deepcopy(model.state_dict())
BEST_ACC = 0.0

def train(model, optimizer, scheduler, data_loader, epoch):
    # Set Model to Training Mode
    scheduler.step()
    model.train(True)

    # Initialize Loss and Score Variables
    running_closs = 0.0
    running_dloss = 0.0
    running_c_acc = 0.0
    running_d_acc = 0.0

    for i, data in enumerate(tqdm(data_loader)):
        # Initialize Data Variables
        image = Variable(data[0], requires_grad=True)
        body  = Variable(data[1], requires_grad=True)
        disc  = Variable(data[2], requires_grad=False)
        cont  = Variable(data[3], requires_grad=False)

        # Utilize CUDA
        if params.USE_CUDA:
            body, image, disc, cont = body.cuda(), image.cuda(), disc.cuda(), cont.cuda()

        # Zero out parameter gradients
        optimizer.zero_grad()

        # Forward Function
        disc_pred, cont_pred = model(body, image)

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()

    # Loss Value Variables
    train_dloss = 0
    train_closs = 0

    # Batch Size Counter
    bcnt = 0

    # Best Tracker
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training Minibatch Iteration
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
        disc_loss = nn.MultiLabelMarginLoss()
        # cont_loss = nn.KLDivLoss()
        # disc_loss = emotic.DiscreteLoss()

        # Compute Test Loss
        d_loss = disc_loss(disc_pred, disc)
        # c_loss = cont_loss(cont_pred, cont.float())

        # Backprop
        # d_loss.backward(retain_graph=True)
        # c_loss.backward(retain_graph=True)
        d_loss.backward()

        optimizer.step()

        # Record Accuracy and Loss Data
        train_dloss += d_loss.data[0]
        # train_closs += c_loss.data[0]

        # Update Log Dashboard Data Point
        if params.VISTORCH_LOG:
            mlog.update_loss(d_loss.data, meter='Discrete Loss')

        # Update Batch Counter
        bcnt += 1

    # Report Loss
    # print('[TRAIN LOSS]\tD_LOSS: ' + str(train_dloss/float(bcnt)) + '\tC_LOSS: ' + str(train_closs/float(bcnt)))
    print('[TRAIN LOSS]\tD_LOSS: ' + str(train_dloss/float(bcnt)))

    if params.VISTORCH_LOG:
        mlog.print_meter(mode="Train", iepoch=epoch)
        mlog.reset_meter(mode="Train", iepoch=epoch)

    # Persist Loss to Log
    # util.loss_log.write(params.LOSS_LOG_DIR + '/emotic_basline_train_loss.csv', str(epoch) + ',' + str(train_dloss/float(bcnt)) + ',' + str(train_closs/float(bcnt)))
    util.loss_log.write(params.LOSS_LOG_DIR + '/emotic_baseline_train_loss.csv', str(epoch) + ',' + str(train_dloss/float(bcnt)))

    # Intermediate Model Persistence Routine
    if epoch % params.SAVE_FREQ == 0:
        print('>> MODEL CHECKPOINT PERSISTED')
        emotic.save_model(model, params.MODEL_DIR + '/emotic_baseline_' + str(epoch) + '_' + str(params.TRAIN_EPOCH) + '.pth')

def valid_epoch(epoch, args, model, data_loader):
    model.train(False)
    model.eval()

    # Loss Value Variables
    valid_dloss = 0
    valid_closs = 0

    # Batch Size Counter
    bcnt = 0

    # Validation Minibatch Iteration
    for i, data in enumerate(tqdm(data_loader)):
        # Initialize Data Variables
        image = Variable(data[0], requires_grad=False)
        body  = Variable(data[1], requires_grad=False)
        disc  = Variable(data[2], requires_grad=False)
        cont  = Variable(data[3], requires_grad=False)

        # Utilize CUDA
        if params.USE_CUDA:
            body, image, disc, cont = body.cuda(), image.cuda(), disc.cuda(), cont.cuda()

        # Generate Predictions
        disc_pred, cont_pred = model(body, image)

        # Initialize Weighted Loss Functions
        disc_loss = nn.MultiLabelMarginLoss()
        # cont_loss = nn.MSELoss()

        # Compute Test Loss
        d_loss = disc_loss(disc_pred, disc)
        # c_loss = cont_loss(cont_pred, cont.float())

        # Record Accuracy and Loss Data
        valid_dloss += d_loss.data[0]
        # valid_closs += c_loss.data[0]

        # Update Batch Counter
        bcnt += 1

    # Report Loss
    # print('[VALID LOSS]\tD_LOSS: ' + str(valid_dloss/float(bcnt)) + '\tC_LOSS: ' + str(valid_closs/float(bcnt)))
    print('[VALID LOSS]\tD_LOSS: ' + str(valid_dloss/float(bcnt)))

    # Persist Loss to Log
    # util.loss_log.write(params.LOSS_LOG_DIR + '/emotic_basline_valid_loss.csv', str(epoch) + ',' + str(valid_dloss/float(bcnt)) + ',' + str(valid_closs/float(bcnt)))
    util.loss_log.write(params.LOSS_LOG_DIR + '/emotic_basline_valid_loss.csv', str(epoch) + ',' + str(valid_dloss/float(bcnt)))

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
    normalize = transforms.Normalize(mean=params.EMOTIC_MEAN, std=params.EMOTIC_STD)
    transform = transforms.Compose([transforms.Scale(params.IM_DIM), transforms.ToTensor(), normalize])

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
    emotic.model_summary(model)

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
    optimizer = optim.Adadelta(model.parameters(), lr=params.TRAIN_LR)

    # Initialize Best Model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Perfom Training Iterations
    for epoch in range(params.START_EPOCH, params.TRAIN_EPOCH+1):
        print('EPOCH '+str(epoch)+'/'+str(params.TRAIN_EPOCH))
        train_epoch(epoch, None, model, train_loader, optimizer)
        valid_epoch(epoch, None, model, valid_loader)

    # Save Model
    emotic.save_model(model, params.MODEL_DIR + '/emotic_baseline.pth')
