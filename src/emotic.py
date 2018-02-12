'''
EMOTIC CNN: Baseline Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import params
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

''' Model '''
class EmoticCNN(nn.Module):
    def __init__(self):
        super(EmoticCNN, self).__init__()
        
        # Initialize VGG16 Model
        vgg16 = torchvision.models.vgg16(pretrained=True)

        # Setup Feature Channels
        self.body_channel = vgg16.features
        self.image_channel = vgg16.features

        # Average Fusion Layers
        self.avg_pool_body = nn.AvgPool2d(4, stride=1)
        self.avg_pool_imag = nn.AvgPool2d(3, stride=16)
        
        # Feature Flatten Layers
        self.flat_body = Flatten()
        self.flat_imag = Flatten()

        # Fully Connected Layers
        # FIX: Reconsider tensor shape along each transformation.
        self.bn_layer = nn.BatchNorm2d(13312)
        self.fc_layer = nn.Linear(13312, 256)

        # Output Layers
        self.discrete_out = nn.Linear(256, 26)
        self.vad_out = nn.Linear(256, 3)

    def forward(self, body, image):
        # VGG16 Feature Extraction Channels
        x_1 = self.body_channel(body)
        x_2 = self.image_channel(image)

        # Global Average Pooling
        x_1 = self.avg_pool_body(x_1)
        x_2 = self.avg_pool_imag(x_2)
        
        # Flatten Layers
        x_1 = self.flat_body(x_1)
        x_2 = self.flat_imag(x_2)

        # Concat + FC Layer
        out = torch.cat([x_1, x_2], 1)
        out = self.bn_layer(out)
        out = self.fc_layer(out)

        # Output Layers
        y_1 = F.softmax(self.discrete_out(out))
        y_2 = self.vad_out(out)
        
        return y_1, y_2

# Auxillary Flatten Function
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def model_summary(model):
    for m in model.modules(): print(m)

''' Loss Function '''
class DiscreteLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiscreteLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        if self.weight:
            disc_w = torch.ones(params.NDIM_DISC)
        else:
            sum_class = torch.sum(target, dim=0).float()
            mask = sum_class > 0.5

            if params.USE_CUDA:
                prev_w = torch.FloatTensor(params.NDIM_DISC).cuda() / torch.log(sum_class.data.float() + params.LDISC_C)
            else:
                prev_w = torch.FloatTensor(params.NDIM_DISC) / torch.log(sum_class.data.float() + params.LDISC_C)
            
            disc_w = mask.data.float() * prev_w

        # Compute Weighted Loss
        N = input.size()[0]
        loss = torch.sum((input.data - target.data.float()).pow(2), dim=0) / N
        w_loss = torch.sum(loss * disc_w, dim=0)
        
        # Return Loss Back as Torch Tensor
        return Variable(w_loss, requires_grad=True)

def disc_weight(input, target, weight=None):
    if weight == 'ONES':
        disc_w = torch.ones(params.NDIM_DISC)
    else:
        sum_class = torch.sum(target, dim=0).float()
        mask = sum_class > 0.5

        if params.USE_CUDA:
            prev_w = torch.FloatTensor(params.NDIM_DISC).cuda() / torch.log(sum_class.data.float() + params.LDISC_C)
        else:
            prev_w = torch.FloatTensor(params.NDIM_DISC) / torch.log(sum_class.data.float() + params.LDISC_C)

        disc_w = mask.data.float() * prev_w
    
    return disc_w

class ContinuousLoss(nn.Module):
    def __init__(self, weight=None):
        super(ContinuousLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # Compute Weight Values
        if self.weight == None: self.weight = cont_weight(input, target)
        
        # Compute Weighted Loss
        N = input.size()[0]
        loss = torch.sum((input.data - target.data.float()).pow(2)) / N
        w_loss = torch.sum(loss * self.weight, dim=0)

        # Return Loss Back as Torch Tensor
        return Variable(w_loss, requires_grad=True)

def cont_weight(input, target, weight=None):
    if weight == 'ONES':
        cont_w = torch.ones(params.NDIM_CONT)
    else:
        diff = torch.sum(torch.abs(input.data - target.data).float(), dim=0) / input.data.size()[0]
        if params.USE_CUDA:
            cont_w = diff > torch.FloatTensor(params.NDIM_CONT).fill_(params.LOSS_CONT_MARGIN).cuda()
        else:
            cont_w = diff > torch.FloatTensor(params.NDIM_CONT).fill_(params.LOSS_CONT_MARGIN)
    
    return cont_w.float()

if __name__ == '__main__':
    # Discrete Loss Function Test
    '''
    loss = DiscreteLoss()
    
    y_pred = torch.LongTensor([[1, 0, 1], [0, 1, 1]]).cuda()
    y_real = torch.LongTensor([[0, 1, 1], [1, 1, 0]]).cuda()

    out = loss(y_pred, y_real)
    print(out)
    '''

    # Continuous Loss Function Test
    y_pred = torch.FloatTensor([[3, 7, 5], [2, 7, 8]]).cuda() / 10.0
    y_real = torch.FloatTensor([[4, 2, 3], [3, 4, 9]]).cuda() / 10.0

    loss = ContinuousLoss()
    
    out = loss(y_pred, y_real)
    print(out)
