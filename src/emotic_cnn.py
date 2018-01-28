'''
EMOTIC CNN [Keras Version]
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
from keras import Input
from keras.models import Model
from keras.layers import Average, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16 as VGG16

def build_model():
    # Input Channels
    input_1 = Input(shape=(256, 256, 3, ), name='body')
    input_2 = Input(shape=(256, 256, 3, ), name='image')
    
    ## Channel 1: Body Feature Extraction
    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='body_blk1_cnn1')(input_1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='body_blk1_cnn2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='body_blk1_pool1')(x1)

    # Block 2
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='body_blk2_cnn1')(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='body_blk2_cnn2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='body_blk2_pool1')(x1)

    # Block 3
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='body_blk3_cnn1')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='body_blk3_cnn2')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='body_blk3_cnn3')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='body_blk3_pool1')(x1)

    # Block 4
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk4_cnn1')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk4_cnn2')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk4_cnn3')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='body_blk4_pool1')(x1)

    # Block 5
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk5_cnn1')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk5_cnn2')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='body_blk5_cnn3')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='body_blk5_pool1')(x1)

    ## Channel 2: Context Feature Extraction
    # Block 1
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='ctx_blk1_cnn1')(input_2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='ctx_blk1_cnn2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ctx_blk1_pool1')(x2)

    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='ctx_blk2_cnn1')(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='ctx_blk2_cnn2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ctx_blk2_pool1')(x2)

    # Block 3
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='ctx_blk3_cnn1')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='ctx_blk3_cnn2')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='ctx_blk3_cnn3')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ctx_blk3_pool1')(x2)

    # Block 4
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk4_cnn1')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk4_cnn2')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk4_cnn3')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ctx_blk4_pool1')(x2)

    # Block 5
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk5_cnn1')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk5_cnn2')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='ctx_blk5_cnn3')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ctx_blk5_pool1')(x2)

    # Fusion Layer
    avg_fusion = Average()([x1, x2])
    dense_layer = Dense(256, activation='relu')(avg_fusion)

    # Output Layer
    disc = Dense(26, activation='softmax')(dense_layer)
    cont = Dense(26, activation='linear')(dense_layer)

    # Model Initialization
    model = Model([input_1, input_2], [disc, cont])
    return model
