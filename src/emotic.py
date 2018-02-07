'''
EMOTIC CNN: Baseline Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import params

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

''' Model '''
class EmoticCNN(nn.Module):
    def __init__(self):
        super(EmoticCNN, self).__init__()
        
        # Initialize VGG16 Model
        vgg16 = torchvision.models.vgg16(pretrained=True)

        # Setup Feature Channels
        self.body_channel = vgg16.features
        self.image_channel = vgg16.features

        # Average Fusion and Hidden Layers
        self.avg_pool = nn.AvgPool2d(3, stride=2)
        self.fc_layer = nn.ReLU(256)        

    def forward(self, body, image):
        x_1 = self.body_channel(body)
        x_2 = self.image_channel(image)
        
        out = self.avg_pool(torch.cat([x_1, x_2], 1))
        out = self.fc_layer(out)

        y_1 = F.softmax(out, 26)
        y_2 = F.linear(out, 3)

        return y_1, y_2

def model_summary(model):
    for m in model.modules(): print(m)

''' Loss Function '''
# TODO: Experiment with various loss functions to empirically observe and compare results.

class DiscreteLoss(nn.Module):
    # TODO: Check how data formatting is being performed during backpropagation
    def __init__(self, weight=None):
        super(DiscreteLoss, self).__init__()
    
    def forward(self, input, target):
        dim = input.dim()
        print(dim)

        # Compute Batch Weights
        if weight:
            weight = Variable(weight)
        elif weight is None:
            sum_class = torch.sum(A, dim=0)
            mask = sum_classes > 0.5
            prev_w = Variable(torch.ones(params.NDIM_DISC)) / torch.log(sum_class + params.LDSIC_C)
            disc_w = mask * prev_w
