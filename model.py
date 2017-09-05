from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.utils.conv_utils import conv_output_length
from keras.layers import GaussianNoise

'''
This file builds the models


'''

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape,Flatten, Lambda, Input,\
    Masking, Convolution1D, BatchNormalization, GRU, Conv1D, RepeatVector, Conv2D
from keras.optimizers import SGD, adam
from keras.layers import ZeroPadding1D, Convolution1D, ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add  # , # concatenate BAD FOR COREML
from keras.utils.conv_utils import conv_output_length
from keras.activations import relu

import tensorflow as tf

def selu(x):
    # from Keras 2.0.6 - does not exist in 2.0.4
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
       x: A tensor or variable to compute the activation function for.
    # References
       - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)

def clipped_relu(x):
    return relu(x, max_value=20)

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred

######################################

######################################

def ds1_dropout(input_dim=26, fc_size=2048, rnn_size=512, dropout=[0.1, 0.1, 0.1], output_dim=29):
    """ DeepSpeech 1 Implementation with Dropout

    Architecture:
        Input MFCC TIMEx26
        3 Fully Connected using Clipped Relu activation function
        3 Dropout layers between each FC
        1 BiDirectional LSTM
        1 Dropout applied to BLSTM
        1 Dropout applied to FC dense
        1 Fully connected Softmax

    Details:
        - Uses MFCC's rather paper's 80 linear spaced log filterbanks
        - Uses LSTM's rather than SimpleRNN
        - No translation of raw audio by 5ms
        - No stride the RNN

    Reference:
        https://arxiv.org/abs/1412.5567
    """
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})
    K.set_learning_phase(1)

    # Creates a tensor there are usually 26 MFCC
    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, max_batch_seq, 26)

    # First 3 FC layers
    init = random_normal(stddev=0.046875)
    x = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)

    # Layer 4 BiDirectional RNN
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu, dropout=dropout[1],
                                kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(x)


    # Layer 5+6 Time Dist Dense Layer & Softmax
    # x = TimeDistributed(Dense(fc_size, activation=clipped_relu, kernel_initializer=init, bias_initializer=init))(x)
    x = TimeDistributed(Dropout(dropout[2]))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"), name="out")(x)

    # Change shape
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model


def ds1(input_dim=26, fc_size=1024, rnn_size=1024, output_dim=29):
    """ DeepSpeech 1 Implementation without dropout

    Architecture:
        Input MFCC TIMEx26
        3 Fully Connected using Clipped Relu activation function
        1 BiDirectional LSTM
        1 Fully connected Softmax

    Details:
        - Removed Dropout on this implementation
        - Uses MFCC's rather paper's 80 linear spaced log filterbanks
        - Uses LSTM's rather than SimpleRNN
        - No translation of raw audio by 5ms
        - No stride the RNN

    References:
        https://arxiv.org/abs/1412.5567
    """
    # hack to get clipped_relu to work on bidir layer
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})


    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)

    init = random_normal(stddev=0.046875)

    # First 3 FC layers
    x = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)


    # # Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu,
                                kernel_initializer='glorot_uniform', name='birnn'), merge_mode='sum')(x)  #

    # Layer 5+6 Time Dist Layer & Softmax

    # x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"), name="out")(x)
    #y_pred = Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax")(x)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])


    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model


def ds2_gru_model(input_dim=161, fc_size=1024, rnn_size=512, output_dim=29, initialization='glorot_uniform',
                  conv_layers=1, gru_layers=1, use_conv=True):
    """ DeepSpeech 2 implementation

    Architecture:
        Input Spectrogram TIMEx161
        1 Batch Normalisation layer on input
        1-3 Convolutional Layers
        1 Batch Normalisation layer
        1-7 BiDirectional GRU Layers
        1 Batch Normalisation layer
        1 Fully connected Dense
        1 Softmax output

    Details:
       - Uses Spectrogram as input rather than MFCC
       - Did not use BN on the first input
       - Network does not dynamically adapt to maximum audio size in the first convolutional layer. Max conv
          length padded at 2048 chars, otherwise use_conv=False

    Reference:
        https://arxiv.org/abs/1512.02595
    """

    K.set_learning_phase(1)

    input_data = Input(shape=(None, input_dim), name='the_input')
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)

    if use_conv:
        conv = ZeroPadding1D(padding=(0, 2048))(x)
        for l in range(conv_layers):
            x = Conv1D(filters=fc_size, name='conv_{}'.format(l+1), kernel_size=11, padding='valid', activation='relu', strides=2)(conv)
    else:
        for l in range(conv_layers):
            x = TimeDistributed(Dense(fc_size, name='fc_{}'.format(l + 1), activation='relu'))(x)  # >>(?, time, fc_size)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

    for l in range(gru_layers):
        x = Bidirectional(GRU(rnn_size, name='fc_{}'.format(l + 1), return_sequences=True, activation='relu', kernel_initializer=initialization),
                      merge_mode='sum')(x)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

    # Last Layer 5+6 Time Dist Dense Layer & Softmax
    x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)

    # labels = K.placeholder(name='the_labels', ndim=1, dtype='int32')
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model


def ownModel(input_dim=26, fc_size=512, rnn_size=512, dropout=[0.1, 0.1, 0.1], output_dim=29):
    """ Own model BN+SELU-FC+GRU+BN+DR

    Architecture:
        Batch Normalisation layer on the input data
        1 Fully connected layer of fc_size with SELU
        2 Fully connected layer of fc_size with Clipped Relu
        3 Dropout layers applied between the FC layers
        Batch Normalisation layer on the final FC output
        1 BiDirectional GRU layer with Clipped Relu
        1 Fully connected layer of fc_size with SELU
        1 Dropout layer
        1 Softmax out


    """
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    K.set_learning_phase(1)

    # Creates a tensor there are usually 26 MFCC
    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, max_batch_seq, 26)

    x = BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3,center=True,scale=True)(input_data)

    # First 3 FC layers
    init = random_normal(stddev=0.046875)
    x = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=selu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

    # Layer 4 BiDirectional RNN
    x = Bidirectional(GRU(rnn_size, return_sequences=True, activation=clipped_relu, dropout=dropout[1],
                                kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(x)

    # Layer 5+6 Time Dist Dense Layer & Softmax
    x = TimeDistributed(Dense(fc_size, activation=selu, kernel_initializer=init, bias_initializer=init))(x)
    x = TimeDistributed(Dropout(dropout[2]))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"), name="out")(x)

    # Change shape
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model


def graves(input_dim=26, rnn_size=512, output_dim=29, std=0.6):
    """ Implementation of Graves 2006 model

    Architecture:
        Gaussian Noise on input
        BiDirectional LSTM

    Reference:
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    # x = BatchNormalization(axis=-1)(input_data)

    x = GaussianNoise(std)(input_data)
    x = Bidirectional(LSTM(rnn_size,
                      return_sequences=True,
                      implementation=0))(x)
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax'))(x)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])


    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model

def cnn_city(input_dim=161, fc_size=1024, rnn_size=512, output_dim=29, initialization='glorot_uniform',
                  conv_layers=4):
    """ Pure CNN implementation

    Architecture:

        1 Convolutional Layers

        1 Fully connected Dense
        1 Softmax output

    Details:s
       - Network does not dynamically adapt to maximum audio size in the first convolutional layer. Max conv
          length padded at 2048 chars, otherwise use_conv=False

    Reference:

    """

    #filters = outputsize
    #kernal_size = heigth and width of conv window
    #strides = stepsize on conv window

    kernel_size = 11  #
    conv_depth_1 = 64  #
    conv_depth_2 = 256  #

    input_data = Input(shape=(None, input_dim), name='the_input') #batch x time x spectro size
    conv = ZeroPadding1D(padding=(0, 2048))(input_data) #pad on time dimension

    x = Conv1D(filters=128, name='conv_1', kernel_size=kernel_size, padding='valid', activation='relu', strides=2)(conv)
    # x = Conv1D(filters=1024, name='conv_2', kernel_size=kernel_size, padding='valid', activation='relu', strides=2)(x)


    # Last Layer 5+6 Time Dist Dense Layer & Softmax
    x = TimeDistributed(Dense(fc_size, activation='relu'))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)

    # labels = K.placeholder(name='the_labels', ndim=1, dtype='int32')
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model




def const(input_dim=26, fc_size=1024, rnn_size=1024, output_dim=29):
    """ Implementation of constrained model for CoreML

    Architecture:
        N number of Fully connected layer of variable FC units
        *optional* GRU RNN of rnn_size

    Details:
        The RNN has been removed in order to allow the network to run in coreml

    """

    #loop FC
    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, time, input_dim)
    x = input_data
    init = random_normal(stddev=0.046875)

    layercount = 3
    for l in range(layercount):
        x = TimeDistributed(Dense(fc_size, name='fc_{}'.format(l+1), kernel_initializer=init,
                                  bias_initializer=init, activation='relu'))(x)  # >>(?, time, fc_size)

    # x = GRU(rnn_size, return_sequences=True, activation='relu', name='rnn1')(x)  # >> (?, time, rnn_size)

    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)  # >> (?,time,output_dim)


    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])


    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model


###########################
# TRANSFER MODEL WEIGHTS

def build_const_no_ctc_and_xfer_weights(loaded_model, input_dim=26, fc_size=1024, rnn_size=512,
                         output_dim=29):
    '''
    CONST model but convert into CoreML
    '''



    K.set_learning_phase(0)

    for ind, i in enumerate(loaded_model.layers):
        print(ind, i)

    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)

    x = input_data

    layercount = 3
    for l in range(layercount):
        x = TimeDistributed(Dense(fc_size, name='fc_{}'.format(l+1), weights=loaded_model.layers[l+1].get_weights(),
                                  activation='relu'))(x)  # >>(?, time, fc_size)
    # x = GRU(rnn_size, return_sequences=True, activation='relu', name='rnn1',
    #         weights=loaded_model.layers[4].get_weights())(x)  # >> (?, time, rnn_size)

    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax",
                                   weights=loaded_model.layers[5].get_weights()))(x)  # >> (?,time,output_dim)






    # First 3 FC layers
    # x = TimeDistributed(Dense(fc_size, name='fc1', activation='relu',
    #                           weights=loaded_model.layers[1].get_weights()))(input_data)
    #
    # x = TimeDistributed(Dense(fc_size, name='fc2', activation='relu',
    #                           weights=loaded_model.layers[2].get_weights()))(x)  # >>(?, 778, 2048)
    #
    # x = TimeDistributed(Dense(fc_size, name='fc3', activation='relu',
    #                           weights=loaded_model.layers[3].get_weights()))(x)  # >>(?, 778, 2048)
    # conv = ZeroPadding1D(padding=(1, 1000))(input_data)
    # conv = Conv1D(filters=2, kernel_size=10, padding='valid', activation='relu',
    #                                weights=loaded_model.layers[2].get_weights(), strides=2)(conv)


    # Layer 4 RNN
    # rnn_1 = GRU(rnn_size, return_sequences=True, activation='relu', name='rnn1',
    #                 weights=loaded_model.layers[3].get_weights())(conv)
    # # rnn_2 = GRU(rnn_size, return_sequences=True, activation='relu', name='rnn2',
    # #             weights=loaded_model.layers[5].get_weights())(rnn_1)
    #
    # x = Dense(fc_size, activation='relu',
    #           weights=loaded_model.layers[5].get_weights())(rnn_1)
    # x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation='relu'),
    #                   weights=loaded_model.layers[3].get_weights(),
    #                   merge_mode='sum')(conv)

    # conv = ZeroPadding1D(padding=(1, 500))(rnn_1)
    # conv = Convolution1D(1, 2, padding='valid',
    #                      weights=loaded_model.layers[6].get_weights())(conv)

    # y_pred = GRU(output_dim, return_sequences=True, activation='softmax', kernel_initializer='glorot_uniform', name='y_pred',
    #             weights=loaded_model.layers[4].get_weights())(x)

    # x = TimeDistributed(Dense(fc_size, activation='relu',
    #                 weights=loaded_model.layers[5].get_weights()))(rnn_1)
    # y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax",
    #                 weights=loaded_model.layers[6].get_weights()), name="out")(x)
    # y_pred = Dense(output_dim, name="y_pred", activation="softmax",
    #                weights=loaded_model.layers[6].get_weights())(x)


    # Layer 5+6 Time Dist Layer & Softmax
    # y_pred = Dense(output_dim, name="y_pred", activation="softmax", weights=loaded_model.layers[5].get_weights())(rnn_1)

    # y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation='softmax',
    #                                weights=loaded_model.layers[4].get_weights()))(x)

    model = Model(inputs=input_data, outputs=y_pred)
    return model




def build_ds0_no_ctc_and_xfer_weights(loaded_model, input_dim=26, fc_size=1024, rnn_size=512,
                         dropout=[0, 0, 0],
                         output_dim=29):
    '''
    DS1 model but convert into CoreML
    '''

    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})


    K.set_learning_phase(0)

    for ind, i in enumerate(loaded_model.layers):
        print(ind, i)

    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)

    # First 3 FC layers
    x = TimeDistributed(Dense(fc_size, name='fc1', activation='relu',
                              weights=loaded_model.layers[1].get_weights()))(input_data)
    # x = TimeDistributed(Dropout(dropout[0]))(x) #2

    x = TimeDistributed(Dense(fc_size, name='fc2', activation='relu',
                              weights=loaded_model.layers[3].get_weights()))(x)  # >>(?, 778, 2048)
    # x = TimeDistributed(Dropout(dropout[0]))(x) #4

    x = TimeDistributed(Dense(fc_size, name='fc3', activation='relu',
                              weights=loaded_model.layers[5].get_weights()))(x)  # >>(?, 778, 2048)
    # x = TimeDistributed(Dropout(dropout[0]))(x) #6

    # x = Dense(fc_size, name='fc1', activation='relu',
    #                           weights=loaded_model.layers[1].get_weights())(input_data)  # >>(?, 778, 2048)
    # x = Dense(fc_size, name='fc2', activation='relu',
    #                           weights=loaded_model.layers[2].get_weights())(x)  # >>(?, 778, 2048)
    # x = Dense(fc_size, name='fc3', activation='relu',
    #                           weights=loaded_model.layers[3].get_weights())(x)  # >>(?, 778, 2048)



    # Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation='relu',
                                kernel_initializer='he_normal'),
                      weights=loaded_model.layers[7].get_weights(),
                                merge_mode='sum')(x)


    x = TimeDistributed(Dense(fc_size, activation='relu',
                    weights=loaded_model.layers[8].get_weights()))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax", weights=loaded_model.layers[10].get_weights()), name="out")(x)

    # Layer 5+6 Time Dist Layer & Softmax
    #y_pred = Dense(num_classes, name="y_pred", activation="softmax", weights=loaded_model.layers[4].get_weights())(x)

    model = Model(inputs=input_data, outputs=y_pred)
    return model


def build_ds5_no_ctc_and_xfer_weights(loaded_model, input_dim=161, fc_size=1024, rnn_size=512, output_dim=29, initialization='glorot_uniform',
                  conv_layers=4):
    """ Pure CNN implementation"""


    K.set_learning_phase(0)
    for ind, i in enumerate(loaded_model.layers):
        print(ind, i)

    kernel_size = 11  #
    conv_depth_1 = 64  #
    conv_depth_2 = 256  #

    input_data = Input(shape=(None, input_dim), name='the_input') #batch x time x spectro size
    conv = ZeroPadding1D(padding=(0, 2048))(input_data) #pad on time dimension

    x = Conv1D(filters=128, name='conv_1', kernel_size=kernel_size, padding='valid', activation='relu', strides=2,
            weights = loaded_model.layers[2].get_weights())(conv)
    # x = Conv1D(filters=1024, name='conv_2', kernel_size=kernel_size, padding='valid', activation='relu', strides=2,
    #            weights=loaded_model.layers[3].get_weights())(x)


    # Last Layer 5+6 Time Dist Dense Layer & Softmax
    x = TimeDistributed(Dense(fc_size, activation='relu',
                              weights=loaded_model.layers[3].get_weights()))(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)

    model = Model(inputs=input_data, outputs=y_pred)

    return model

