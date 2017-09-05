
import pandas as pd
import argparse
import socket
import os

''' script to limit dataset to 140char per transcript and make the pandas files used for DeepSpeech same as kDS'''

def main(args):

    csvslist = args.train_files



    for csvpath in csvslist.split(','):
        print("Reading csv:",csvpath)

        if os.path.isfile(csvpath):
            df_new = pd.read_csv(csvpath, sep=',')
            df_trim = df_new[df_new['transcript'].map(len) <= 140]
            trimpath = csvpath[:-4]+"_trim.csv"
            print("writing to path:", trimpath)
            df_trim.to_csv(trimpath, sep=',', header=True, index=False)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--train_files', type=str, default='',
                       help='list of all train files, seperated by a comma if multiple')
    parser.add_argument('--valid_files', type=str, default='',
                        help='')

    args = parser.parse_args()

    #detect if local user here

    timit_path = "../data/LDC/timit/"
    libri_path = "../data/LibriSpeech/"
    ted_path = "../data/ted/"

    sep = ","
    args.train_files = timit_path + "timit_train.csv"+ sep + \
                        libri_path + "librivox-dev-clean.csv"+ sep + \
                        ted_path + "ted-dev.csv"

    args.valid_files = timit_path+"timit_test.csv"

    print(args)
    main(args)
