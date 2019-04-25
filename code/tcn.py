"""
Temporal Convolutional Network (TCN) implemented in Keras
This implementation is based on the original paper of Bai Shaojie, Kolter J Zico and Koltun Vladlen.
# References
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](http://arxiv.org/abs/1803.01271)
"""

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation, Add, SpatialDropout1D, Masking, BatchNormalization
from keras import regularizers, initializers
from custom_layers import CausalConv1D
import numpy as np


def TCN(input_shape=None, residual_blocks=3, tcn_layers=-1, filters=12, filters_size=3,
        dropout_rate=None, weight_decay=1e-4, depth=40, seed=0, masking=False, mask_value=-10.0):
    """
        Creating a Temporal Convolutional Network. Supports masking.

        Arguments:
            input_shape     : array-like, shape of the input (height, width, depth)
            residual_blocks : amount of residual blocks that will be created (default: 3)
            tcn_layers      : number of layers in each residual block. You can also use a list for numbers of layers [2,4,3]
                              or define only 2 to add 2 layers at all residual blocks. -1 means that dense_layers will be calculated
                              by the given depth (default: -1)
            filters         : filters per layer
            filters_size    : filter size per layer
            dropout_rate    : defines the dropout rate that is accomplished after each conv layer (except the first one).
            weight_decay    : weight decay of L2 regularization on weights (default: 1e-4)
            depth           : number or layers (default: 40)
            seed            : rng seed
            masking         : whether to use masking. If True, mask value is -10.0

        Returns:
            Model        : A Keras model instance
    """

    if type(tcn_layers) is list:
        if len(tcn_layers) != residual_blocks:
            raise AssertionError(
                'Number of residual blocks have to be same length to specified layers')
    elif tcn_layers == -1:
        tcn_layers = depth // residual_blocks
        tcn_layers = [tcn_layers for _ in range(residual_blocks)]
    else:
        tcn_layers = [tcn_layers for _ in range(residual_blocks)]

    if type(filters) is list:
        if type(tcn_layers) is list:
            for i, layer_size in enumerate(tcn_layers):
                if layer_size != len(filters[i]):
                    raise AssertionError(
                        'Number of filters have to be same to layers. Found filters {}, layers {}'.format(np.sum(np.array(filters).shape), np.sum(tcn_layers)))
        elif np.prod(np.array(filters).shape) != np.sum(tcn_layers):
            raise AssertionError(
                'Number of filters have to be same to layers. Found filters {}, layers {}'.format(np.sum(np.array(filters).shape), np.sum(tcn_layers)))
    else:
        filters = [[filters for _ in range(tcn_layers[i])] for i in range(residual_blocks)]

    if type(filters_size) is list:
        if np.prod(np.array(filters_size).shape) != np.sum(tcn_layers):
            raise AssertionError(
                'Number of filters_size have to be same to layers. Found filters_size {}, layers {}'.format(np.sum(np.array(filters_size).shape), np.sum(tcn_layers)))
    else:
        filters_size = [[filters_size for _ in range(tcn_layers[i])] for i in range(residual_blocks)]

    seq_input = Input(shape=input_shape)
    rf = 1
    nb_dilation = 1

    print('Creating TCN')
    print('#############################################')
    print('Residual blocks: %s' % residual_blocks)
    print('Layers per residual block: %s' % tcn_layers)
    print('Filters per layer: %s' % filters)
    print('Filters size per layer: %s' % filters_size)

    kernel_init = initializers.glorot_normal(seed=seed)
    kernel_regl = regularizers.l2(weight_decay)

    x = seq_input
    if masking:
        x = Masking(mask_value)(seq_input)

    # Building residual blocks
    for block in range(residual_blocks):

        # Add residual block
        x, nb_dilation, rf = residual_block(x, tcn_layers[block], filters[block], filters_size[block], nb_dilation, rf, dropout_rate, kernel_regl, kernel_init)

    print('Last layer receptive field: %s' % rf)
    print('#############################################')

    return Model(seq_input, x, name='tcn')


def residual_block(x, nb_layers, nb_channels, filters, nb_dilation, rf, dropout_rate=None, regularizer=None, initializer='glorot_uniform'):
    """
    Creates a residual block and concatenates inputs
    """
    cb = CausalConv1D(nb_channels[-1], (1,),
                use_bias=True,
                kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    for i in range(nb_layers):
        x, nb_dilation, rf = tcn_block(
            x, nb_channels[i], filters[i], nb_dilation, rf, dropout_rate, regularizer, initializer)
    x = Add()([cb, x])
    return x, nb_dilation, rf


def tcn_block(x, nb_channels, filter_size, nb_dilation, rf, dropout_rate=None, regularizer=None, initializer='glorot_uniform'):
    """
    Creates a convolution block consisting of Conv-ReLU-WeightNorm-Dropout.
    Optional: dropout
    """

    # Standard (Conv-ReLU-WeightNorm-Dropout)
    x = CausalConv1D(nb_channels, filter_size, 
                use_bias=True,
                dilation_rate=nb_dilation,
                kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    x = Activation('relu')(x)

    # Dropout
    if dropout_rate:
        x = SpatialDropout1D(dropout_rate)(x)

    rf = rf + (filter_size-1)*nb_dilation
    nb_dilation *= 2
    return x, nb_dilation, rf
