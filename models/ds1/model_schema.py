
from model_utils import *
import keras
from keras.models import Model
from keras.layers import Dense, Bidirectional, Lambda, Input, LSTM
from keras.layers import TimeDistributed
from keras.activations import relu
from keras.initializers import RandomNormal

print("LOADING MODEL_SCHEMA")


def metadata():
    FOLDER      = "ds1"
    TAG         = "ds1"
    AUTHOR      = "robmsmt"
    DESCRIPTION = "DeepSpeech1 implementation"
    DATASETS    = "Mixed librispeech"


def model(input_dim=26, fc_size=1024, rnn_size=1024, output_dim=29):
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

    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)

    init = RandomNormal(stddev=0.046875)

    # First 3 FC layers
    x = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=relu))(input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=relu))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=relu))(x)  # >>(?, 778, 2048)

    # Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=relu,
                                kernel_initializer='glorot_uniform', name='birnn'), merge_mode='sum')(x)  #

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"), name="out")(x)

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
