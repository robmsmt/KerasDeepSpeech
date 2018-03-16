#!/usr/bin/env python

import os
import argparse
from engine.KerasDeepSpeech import KDS


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='ds1',
                        help='name of model inside models/ folder. Default is ds1')

    data_path = os.path.join(ROOT_DIR, "data")
    parser.add_argument('--train_files', type=str, default='',
                       help='SINGLE CSV file that will be auto split into train/validation/test')
    # parser.add_argument('--valid_files', type=str, default='',
    #                    help='list of all validation CSV files, seperate by a comma if multiple')
    # parser.add_argument('--test_files', type=str, default='',
    #                    help='list of all test CSV files, seperate by a comma if multiple')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the learning rate used by the optimiser')
    parser.add_argument('--opt', type=str, default='adam',
                        help='the optimiser to use, default is adam')
    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=8,
                       help='batch_size used to train the model')

    args = parser.parse_args()

    # 1. initialise model
    models_path = os.path.join(ROOT_DIR, "models")
    model_meta = {'model_name': args.model_name,
                  'models_path': models_path}

    kds = KDS(model_meta)

    # 2. initialise data
    data_meta = {
        "train_files": args.train_files,
        "test_files": args.test_files
    }
    kds.data_init(data_meta)


    # train model

    training_params = {"learning_rate": args.learning_rate,
                       "opt": args.opt,
                       "epochs": args.epochs,
                       "batchsize": args.batchsize,}
    kds.train()


