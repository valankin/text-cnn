import numpy as np
import theano.tensor as T
import keras.backend as K
from keras.layers.core import MaskedLayer


def min_max_pool2d(x):
    max_x =  K.pool2d(x, pool_size=(2, 2), strides=(2, 2))
    min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(2, 2))
    return K.concatenate([max_x, min_x], axis=1) # concatenate on channel

def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] /= 2
    shape[3] /= 2
    return tuple(shape)


class KMaxPooling(MaskedLayer):
    def __init__(self, pooling_size):
        super(MaskedLayer, self).__init__()
        self.pooling_size = pooling_size
        self.input = T.tensor3()

    def get_output_mask(self, train=False):
        return None

    def get_output(self, train=False):
        data = self.get_input(train)
        mask = self.get_input_mask(train)

        if mask is None:
            mask = T.sum(T.ones_like(data), axis=-1)
        mask = mask.dimshuffle(0, 1, "x")

        masked_data = T.switch(T.eq(mask, 0), -np.inf, data)

        result = masked_data[T.arange(masked_data.shape[0]).dimshuffle(0, "x", "x"),
                             T.sort(T.argsort(masked_data, axis=1)[:, -self.pooling_size:, :], axis=1),
                             T.arange(masked_data.shape[2]).dimshuffle("x", "x", 0)]

        return result

    def get_config(self):
        return {"name" : self.__class__.__name__, "pooling_size" : self.pooling_size}