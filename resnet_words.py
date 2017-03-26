"""Train a convnet on the MNIST database with ResNets.

ResNets are a bit overkill for this problem, but this illustrates how to use
the Residual wrapper on ConvNets.

See: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
"""

from __future__ import print_function
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import theano
import theano.tensor as T
import keras.backend as K
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import*
from sklearn.cross_validation import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.callbacks import CSVLogger,ReduceLROnPlateau
from data_loader import get_data
from residual_blocks import residual_block

from numpy import random
random.seed(42)  # @UndefinedVariable

from keras.utils import np_utils
from resnet import Residual


#############################DATA###################################
vectors,x,y,vocabulary,vocabulary_inv,embedding_weights = get_data()

model_variation = 'CNN-non-static'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    if model_variation=='CNN-static':
        x = vectors
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    
print(len(x),len(y))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Vocabulary Size: {:d}".format(len(vocabulary)))

# Model Hyperparameters
sequence_length = x.shape[1]#56
embedding_dim = 300          
vocabulary_size = len(vocabulary_inv)
####################################################################
#########################TRAINING_PARAMETERS########################
val_split = 0.1
batch_size = 64
nb_classes = 2
nb_epoch = 100

img_rows, img_cols = sequence_length, embedding_dim
kernel_size = (3, embedding_dim)
pool_size=(1,sequence_length - kernel_size[0] + 1)
####################################################################

# Model
input_var = Input(shape=(sequence_length,), dtype='int32')

embeddings = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size,input_length=sequence_length,weights=[embedding_weights],trainable=True)(input_var)

reshape = Reshape((1,sequence_length,embedding_dim))(embeddings)

conv1 = Convolution2D(8, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu', init='normal', dim_ordering='th')(reshape)
# conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
conv2 = Convolution2D(8, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu', init='normal', dim_ordering='th')(conv1)

resnet = conv2
for _ in range(2):
    resnet = Residual(Convolution2D(8, kernel_size[0], kernel_size[1],
                                  border_mode='same', init='normal', dim_ordering='th'))(resnet)
    resnet = Activation('relu')(resnet)

mxpool = MaxPooling2D(pool_size=pool_size,strides=(1,1), border_mode='same', dim_ordering='th')(resnet)
flat = Flatten()(mxpool)
dropout = Dropout(0.5)(flat)
softmax = Dense(output_dim=nb_classes, activation='sigmoid')(dropout)

model = Model(input=[input_var], output=[softmax])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#########################CALLBACKS######################################
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
#checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
########################################################################
model.fit(X_train,y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(X_test,y_test),
          callbacks=[reduce_lr])  

#model.save('mnist_model.h5')