#!/usr/bin/env python

'''

KERAS Deep Speech - test script

'''

#####################################################

import argparse
import datetime
import socket

#####################################################

from utils import *
from generator import *
from data import combine_all_wavs_and_trans_from_csvs
from model import *
from report import *

import keras

# from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

#######################################################


def main(args):
    '''

    only args.name args.test_files and args.loadcheckpointpath can be passed as args


    '''

    print("Getting data from arguments")
    test_dataprops, df_test = combine_all_wavs_and_trans_from_csvs(args.test_files, sortagrad=False)


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
    testdata = BatchGenerator(dataframe=df_test, dataproperties=test_dataprops,
                               training=False, batch_size=1, model_input_type=model_input_type)



    ## 3. Load existing or error
    if args.loadcheckpointpath:
        # load existing
        print("Loading model")

        cp = args.loadcheckpointpath
        assert(os.path.isdir(cp))
        trimmed = False

        if trimmed:
            model_path = os.path.join(cp, "TRIMMED_ds_model")
        else:
            model_path = os.path.join(cp, "model")
        # assert(os.path.isfile(model_path))

        model = load_model_checkpoint(model_path)
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        print("Model loaded")

    else:
        # new model
        raise("You need to load an existing trained model")


    model.compile(optimizer=opt, loss=ctc)

    ## 4. test

    train_steps = len(df_test.index) // 200

    try:
        y_pred = model.get_layer('ctc').input[0]
    except Exception as e:
        print("error", e)
        print("couldn't find ctc layer, possibly a trimmed layer, trying other name")
        y_pred = model.get_layer('out').output

    input_data = model.get_layer('the_input').input

    K.set_learning_phase(0)
    report = K.function([input_data, K.learning_phase()], [y_pred])
    report_cb = ReportCallback(report, testdata, model, args.name, save=False)
    report_cb.force_output = True
    report_cb.on_epoch_end(0, logs=None)

    K.clear_session()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--loadcheckpointpath', type=str, default='./checkpoints/trimmed/',
    parser.add_argument('--loadcheckpointpath', type=str, default='./checkpoints/epoch/LER-WER-best-DS3_2017-09-02_13-40',
                       help='If value set, load the checkpoint json '
                            'weights assumed as same name '
                            ' e.g. --loadcheckpointpath ./checkpoints/'
                            'TRIMMED_ds_ctc_model ')
    parser.add_argument('--name', type=str, default='',
                       help='name of run')
    parser.add_argument('--test_files', type=str, default='',
                       help='list of all validation files, seperate by a comma if multiple')

    parser.add_argument('--model_arch', type=int, default=3,
                       help='choose between model_arch versions (when training not loading) '
                            '--model_arch=1 uses DS1 fully connected layers with simplernn'
                            '--model_arch=2 uses DS2 fully connected with GRU'
                            '--model_arch=3 is custom model')

    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if args.name == "":
        args.name = "DS" + str(args.model_arch) + "_" + runtime

    # detect if local user here
    if socket.gethostname().lower() in 'rs-e5550'.lower():

        timit_path = "./data/LDC/timit/"
        libri_path = "./data/LibriSpeech/"
        ted_path = "./data/ted/"
        own_path = "./data/own/"

        # sep = ","
        # args.train_files = timit_path + "timit_train.csv" + sep + \
        #                    libri_path + "librivox-dev-clean.csv" + sep + \
        #                    ted_path + "ted-dev.csv"

        #args.test_files = timit_path + "timit_test.csv"
        args.test_files = own_path + "enron_test.csv"

    assert (keras.__version__ == "2.0.4")  ## CoreML is strict

    print(args)

    main(args)





