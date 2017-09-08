#!/usr/bin/env python

'''

KERAS Deep Speech - end to end speech recognition. Designed for
use with CoreML 0.5.1 to use model on iOS

see conversion scripts

'''

#####################################################

import argparse
import datetime
import os
import socket

import keras
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, Nadam

from data import combine_all_wavs_and_trans_from_csvs
from generator import BatchGenerator
from model import *
from report import ReportCallback
from utils import load_model_checkpoint, save_model, MemoryCallback

#####################################################


#######################################################

# Prevent pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#######################################################


def main(args):
    '''
    There are 5 simple steps to this program
    '''


    #1. combine all data into 2 dataframes (train, valid)
    print("Getting data from arguments")
    train_dataprops, df_train = combine_all_wavs_and_trans_from_csvs(args.train_files, sortagrad=args.sortagrad)
    valid_dataprops, df_valid = combine_all_wavs_and_trans_from_csvs(args.valid_files, sortagrad=args.sortagrad)


    # check any special data model requirments e.g. a spectrogram
    if(args.model_arch == 1):
        model_input_type = "mfcc"
    elif(args.model_arch == 2 or args.model_arch == 5):
        print("Spectrogram required")
        # spectrogram = True
        model_input_type = "spectrogram"
    else:
        model_input_type = "mfcc"


    ## 2. init data generators
    print("Creating data batch generators")
    traindata = BatchGenerator(dataframe=df_train, dataproperties=train_dataprops,
                              training=True, batch_size=args.batchsize, model_input_type=model_input_type)
    validdata = BatchGenerator(dataframe=df_valid, dataproperties=valid_dataprops,
                              training=False, batch_size=args.batchsize, model_input_type=model_input_type)




    output_dir = os.path.join('checkpoints/results',
                                  'model%s_%s' % (args.model_arch,
                                             args.name))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    ## 3. Load existing or create new model
    if args.loadcheckpointpath:
        # load existing
        print("Loading model")

        cp = args.loadcheckpointpath
        assert(os.path.isdir(cp))

        model_path = os.path.join(cp, "model")
        # assert(os.path.isfile(model_path))

        model = load_model_checkpoint(model_path)


        print("Model loaded")
    else:
        # new model recipes here
        print('New model DS{}'.format(args.model_arch))
        if (args.model_arch == 0):
            # DeepSpeech1 with Dropout
            model = ds1_dropout(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size,dropout=[0.1,0.1,0.1], output_dim=29)

        elif(args.model_arch==1):
            # DeepSpeech1 - no dropout
            model = ds1(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch==2):
            # DeepSpeech2 model
            model = ds2_gru_model(input_dim=161, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch==3):
            # own model
            model = ownModel(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, dropout=[0.1, 0.1, 0.1], output_dim=29)

        elif(args.model_arch==4):
            # graves model
            model = graves(input_dim=26, rnn_size=args.rnn_size, output_dim=29, std=0.5)

        elif(args.model_arch==5):
            #cnn city
            model = cnn_city(input_dim=161, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)

        elif(args.model_arch == 6):
            # constrained model
            model = const(input_dim=26, fc_size=args.fc_size, rnn_size=args.rnn_size, output_dim=29)
        else:
            raise("model not found")

        print(model.summary(line_length=80))

        #required to save the JSON
        save_model(model, output_dir)

    if (args.opt.lower() == 'sgd'):
        opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif (args.opt.lower() == 'adam'):
        opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    elif (args.opt.lower() == 'nadam'):
        opt = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    else:
        raise "optimiser not recognised"

    model.compile(optimizer=opt, loss=ctc)

    ## 4. train

    if args.train_steps == 0:
        args.train_steps = len(df_train.index) // args.batchsize
        # print(args.train_steps)
    # we use 1/xth of the validation data at each epoch end to test val score
    if args.valid_steps == 0:

        args.valid_steps = (len(df_valid.index) // args.batchsize)
        # print(args.valid_steps)


    if args.memcheck:
        cb_list = [MemoryCallback()]
    else:
        cb_list = []

    if args.tensorboard:
        tb_cb = TensorBoard(log_dir='./tensorboard/{}/'.format(args.name), write_graph=False, write_images=True)
        cb_list.append(tb_cb)

    y_pred = model.get_layer('ctc').input[0]
    input_data = model.get_layer('the_input').input

    report = K.function([input_data, K.learning_phase()], [y_pred])
    report_cb = ReportCallback(report, validdata, model, args.name, save=True)

    cb_list.append(report_cb)

    model.fit_generator(generator=traindata.next_batch(),
                        steps_per_epoch=args.train_steps,
                        epochs=args.epochs,
                        callbacks=cb_list,
                        validation_data=validdata.next_batch(),
                        validation_steps=args.valid_steps,
                        initial_epoch=0,
                        verbose=1,
                        class_weight=None,
                        max_q_size=10,
                        workers=1,
                        pickle_safe=False
                        )

    # K.clear_session()

    ## These are the most important metrics
    print("Mean WER   :", report_cb.mean_wer_log)
    print("Mean LER   :", report_cb.mean_ler_log)
    print("NormMeanLER:", report_cb.norm_mean_ler_log)

    # export to csv?
    K.clear_session()
#######################################################


if __name__ == '__main__':

    print("Getting args")

    parser = argparse.ArgumentParser()

    parser.add_argument('--tensorboard', type=bool, default=True,
                       help='True/False to use tensorboard')
    parser.add_argument('--memcheck', type=bool, default=False,
                       help='print out memory details for each epoch')
    parser.add_argument('--name', type=str, default='',
                       help='name of run, used to set checkpoint save name. Default uses timestamp')

    parser.add_argument('--train_files', type=str, default='',
                       help='list of all train files, seperated by a comma if multiple')
    parser.add_argument('--valid_files', type=str, default='',
                       help='list of all validation files, seperate by a comma if multiple')

    parser.add_argument('--train_steps', type=int, default=0,
                        help='number of steps for each epoch. Use 0 for automatic')
    parser.add_argument('--valid_steps', type=int, default=0,
                        help='number of validsteps for each epoch. Use 0 for automatic')

    parser.add_argument('--fc_size', type=int, default=512,
                       help='fully connected size for model')
    parser.add_argument('--rnn_size', type=int, default=512,
                       help='size of the rnn')

    parser.add_argument('--loadcheckpointpath', type=str, default='',

                       help='If value set, load the checkpoint in a folder minus name minus the extension '
                            '(weights assumed as same name diff ext) '
                            ' e.g. --loadcheckpointpath ./checkpoints/'
                            'TRIMMED_ds_ctc_model/')

    parser.add_argument('--model_arch', type=int, default=3,
                       help='choose between model_arch versions (when training not loading) '
                            '--model_arch=1 uses DS1 fully connected layers with LSTM'
                            '--model_arch=2 uses DS2 CNN connected with GRU'
                            '--model_arch=3 is Custom model'
                            '--model_arch=4 is Graves 2006 model'
                            '--model_arch=5 is Pure CNN+FC model'
                            '--model_arch=6 is Constrained FC model')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the learning rate used by the optimiser')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='the optimiser to use, default is SGD, ')

    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=2,
                       help='batch_size used to train the model')

    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if args.name == "":
        args.name = "DS"+str(args.model_arch)+"_"+runtime

    #detect user here too
    if args.train_files=="" and args.valid_files=="":
        # if paths to file not specified, assume testing
        # timit_path = "./data/LDC/timit/"
        # libri_path = "./data/LibriSpeech/"
        # ted_path = "./data/ted/"
        # own_path = "./data/own/"
        test_path = "./data/ldc93s1/"
        sep = ","
        args.train_files = test_path + "ldc93s1.csv"
        args.valid_files = test_path + "ldc93s1.csv"
        #args.train_files = libri_path + "librivox-train-clean-100.csv"
        #args.valid_files = libri_path + "librivox-test-clean.csv"


    assert(keras.__version__ == "2.0.4") ## CoreML is strict

    print(args)

    main(args)





