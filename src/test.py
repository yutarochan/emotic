'''
EMOTIC: CNN Model Baseline Testing
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import emotic
import params
import numpy as np
import util.data_csv
from tqdm import tqdm
from util.data_csv import EMOTICData
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from util.mlmetrics import mlmetrics

def test(model, data_loader):
    model.eval()

    # Initialize Parameters and Variables
    count = 0

    DISC_PRED = []
    DISC_ACTU = []
    CONT_PRED = []
    CONT_ACTU = []

    # Testing Iteration Process
    for i, data in enumerate(tqdm(data_loader)):
        # Initialize Data Variables
        image = Variable(data[0], requires_grad=False)
        body  = Variable(data[1], requires_grad=False)
        
        # Utilize CUDA
        if params.USE_CUDA:
            body, image = body.cuda(), image.cuda()
        
        # Perform Inference
        disc_pred, cont_pred = model(body, image)

        # Append Prediction to List
        DISC_PRED.append(disc_pred.cpu().data.numpy().tolist()[0])
        DISC_ACTU.append(data[2].numpy().tolist()[0])
        # CONT_PRED.append(cont_pred.cpu().data.numpy().tolist()[0])
        # CONT_ACTU.append(data[3].numpy().tolist()[0])

        count += 1

    # Compute AP Score for Each Category
    ap_scores = []
    for i in range(26):
        s = roc_auc_score(np.array(DISC_ACTU)[:, i], np.array(DISC_PRED)[:, i] > 0.5, 'weighted')
        ap_scores.append(s)

    for i, j in zip(util.data_csv.cat_name, ap_scores):
        print(i + '\t\t' + str(j))

    print('\nAVG SCORE: ' + str(np.mean(ap_scores)))

    # disc = mlmetrics(np.array(DISC_ACTU), np.array(DISC_PRED))

    # print(disc.accuracy())
    # print(disc.precision())

if __name__ == '__main__':
    print('='*80)
    print('EMOTIC CNN BASELINE MODEL - TESTING')
    print('='*80 + '\n')

    # Load Dataset
    print('-'*80)
    print('Initialize Dataset and Annotations')
    print('-'*80)

    # Data Transformation and Normalization
    normalize = transforms.Normalize(mean=params.EMOTIC_MEAN, std=params.EMOTIC_STD)
    transform = transforms.Compose([transforms.Resize(params.IM_DIM), transforms.ToTensor()])

    # Load Dataset Generator Objects
    test_data  = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_TEST,  transform=transform)
    
    # Initialize Dataset Loader Objects
    test_loader  = DataLoader(test_data, shuffle=False, num_workers=params.NUM_WORKERS)

    # Initialize Model
    print('-'*80)
    print('Initialize Model')
    print('-'*80)

    # model = emotic.EmoticCNN()
    model = emotic.load_model(params.MODEL_DIR+'emotic_baseline_60_150.pth')

    # IF USE_CUDA - Set model parameters compatible against CUDA GPU
    if params.USE_CUDA:
        print('>> CUDA USE ENABLED!')
        print('>> GPU DEVICES AVAILABLE COUNT: ' + str(torch.cuda.device_count()))
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Test Model
    test(model, test_loader)
