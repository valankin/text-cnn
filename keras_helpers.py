from __future__ import print_function
import sys
sys.setrecursionlimit(99999)
import numpy as np
np.random.seed(1337)
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import residual_blocks

batch_size = 32
nb_classes = 2
nb_epoch = 100

def design_for_residual_blocks(num_channel_input=1):
   
    model = Sequential() # it's a CONTAINER, not MODEL
    # set numbers
    num_big_blocks = 3
    height_input = 56
    width_input = 300
    image_patch_sizes = [[3,width_input]]*num_big_blocks
    pool_sizes = [(height_input - 3 + 1, 1)]*num_big_blocks
    n_features = [128, 256, 512, 512, 1024]
    n_features_next = [256, 512, 512, 512, 1024]
    for conv_idx in range(num_big_blocks):    
        n_feat_here = n_features[conv_idx]
        # residual block 0
        model.add(residual_blocks.building_residual_block((num_channel_input, height_input, width_input),
                                                           n_feat_here,
                                                           kernel_sizes=image_patch_sizes[conv_idx]))

        # residual block 1 (you can add it as you want (and your resources allow..))
        if False:
            model.add(residual_blocks.building_residual_block((n_feat_here, height_input, width_input),
                                                               n_feat_here,
                                                               kernel_sizes=image_patch_sizes[conv_idx]))
        
        
        n_feat_next = n_features_next[conv_idx]
        model.add(residual_blocks.building_residual_block((n_feat_here, height_input, width_input),
                                                           n_feat_next,
                                                           kernel_sizes=image_patch_sizes[conv_idx],
                                                           is_subsample=True,
                                                           subsample=pool_sizes[conv_idx]))

        height_input, width_input = model.output_shape[2:]
        num_channel_input = n_feat_next

    print('Average pooling, from (%d,%d) to (1,1)' % (height_input, width_input))
    model.add(AveragePooling2D(pool_size=(height_input, width_input)))
    
    return model

def get_residual_model(img_channels=1, img_rows=56, img_cols=300):
    model = keras.models.Sequential()
    first_layer_channel = 128
    model.add(Convolution2D(first_layer_channel, 3, 300, border_mode='same', input_shape=(img_rows, img_cols,img_channels)))

    model.add(Activation('relu'))
    model.add(design_for_residual_blocks(num_channel_input=first_layer_channel))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model



model = get_residual_model(img_channels=img_channels, img_rows=img_rows, img_cols=img_cols)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

