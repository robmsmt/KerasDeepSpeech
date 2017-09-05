import fnmatch
import os
import pandas as pd
import char_map
from utils import text_to_int_sequence


#######################################################

def clean(word):
    # token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new


def combine_all_wavs_and_trans_from_csvs(csvslist, sortagrad=True, createwordlist=False, delBigTranscripts=True):
    '''Assume that data is in csv already exists with data in form
        path, size, transcript
        this is best approach for loading in moz deepspeech processed files.
    '''

    df_all = pd.DataFrame()

    for csv in csvslist.split(','):
        print("Reading csv:",csv)

        if os.path.isfile(csv):
            try:
                df_new = pd.read_csv(csv, sep=',', encoding='ascii')
            except:
                print("NOT - ASCII, use UTF-8")
                df_new = pd.read_csv(csv, sep=',', encoding='utf-8')
                df_new.transcript.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

            df_all = df_all.append(df_new)

    print("Finished reading in data")

    if delBigTranscripts:
        print("removing any sentences that are too big- tweetsize")
        df_final = df_all[df_all['transcript'].map(len) <= 140]
    else:
        df_final = df_all

    # can output the word list here if required
    if createwordlist:
        df_final['transcript'].to_csv("./lm/df_all_word_list.csv", sep=',', header=False, index=False)  # reorder + out

    listcomb = df_all['transcript'].tolist()
    print("Total number of files:", len(listcomb))


    listcomb = df_final['transcript'].tolist()
    print("Total number of files (after reduction):", len(listcomb))

    comb = []

    for t in listcomb:
        #s = t.decode('utf-8').encode('ascii', errors='ignore')
        comb.append(' '.join(t.split()))

    # print("Train/Test/Valid:",len(train_list_wavs), len(test_list_wavs), len(valid_list_wavs))
    # 6300 TIMIT
    # (4620, 840, 840) TIMIT



    ## SIZE CHECKS
    max_intseq_length = get_max_intseq(comb)
    num_classes = get_number_of_char_classes()

    print("max_intseq_length:", max_intseq_length)
    print("numclasses:", num_classes)

    # VOCAB CHECKS
    all_words, max_trans_charlength = get_words(comb)
    print("max_trans_charlength:", max_trans_charlength)
    # ('max_trans_charlength:', 80)

    ## TODO could readd the mfcc checks for safety
    # ('max_mfcc_len:', 778, 'at comb index:', 541)

    all_vocab = set(all_words)
    print("Words:", len(all_words))
    print("Vocab:", len(all_vocab))

    dataproperties = {
        'target': "timit+librispeech",
        'num_classes': num_classes,
        'all_words': all_words,
        'all_vocab': all_vocab,
        'max_trans_charlength': max_trans_charlength,
        'max_intseq_length': max_intseq_length
    }

    if sortagrad:
        df_final = df_final.sort_values(by='wav_filesize', ascending=True)
    else:
        df_final = df_final.sample(frac=1).reset_index(drop=True)

    #remove mem
    del df_all
    del listcomb

    return dataproperties, df_final


##DATA CHECKS RUN ALL OF THESE

def get_words(comb):
    max_trans_charlength = 0
    all_words = []

    for count, sent in enumerate(comb):
        # count length
        if len(sent) > max_trans_charlength:
            max_trans_charlength = len(sent)
        # build vocab
        for w in sent.split():
            all_words.append(clean(w))

    return all_words, max_trans_charlength

def get_max_intseq(comb):
    max_intseq_length = 0
    for x in comb:
        try:
            y = text_to_int_sequence(x)
            if len(y) > max_intseq_length:
                max_intseq_length = len(y)
        except:
            print("error at:", x)
    return max_intseq_length

def get_number_of_char_classes():
    ## TODO would be better to check with dataset (once cleaned)
    num_classes = len(char_map.char_map)+1 ##need +1 for ctc null char +1 pad
    return num_classes
