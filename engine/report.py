from keras import callbacks
from text import *

import itertools
import numpy as np
import os
import socket
import sys
import keras.backend as K

from utils import save_model, int_to_text_sequence

class ReportCallback(callbacks.Callback):
    def __init__(self, test_func, validdata, model, runtimestr, save):
        self.test_func = test_func

        self.validdata = validdata
        self.validdata_next_val = self.validdata.next_batch()
        self.batch_size = validdata.batch_size
        self.save = save

        # useful if you want to decrease amount in validation
        self.valid_test_devide = 1  # 1=no reduce, 10 = 1/10th
        #if socket.gethostname().lower() in 'rs-e5550'.lower(): self.valid_test_devide = 50

        self.val_best_mean_ed = 0
        self.val_best_norm_mean_ed = 0

        self.lm = get_model()

        self.model = model
        self.runtimestr = runtimestr

        self.mean_wer_log = []
        self.mean_ler_log = []
        self.norm_mean_ler_log = []

        self.earlystopping = True
        self.shuffle_epoch_end = True
        self.force_output = False

    def validate_epoch_end(self, verbose=0):

        originals = []
        results = []
        count = 0
        self.validdata.cur_index = 0  # reset index

        if self.valid_test_devide: #check not zero
            allvalid = (len(self.validdata.wavpath) // self.validdata.batch_size) // self.valid_test_devide


        #make a pass through all the validation data and assess score
        for c in range(0, allvalid):

            word_batch = next(self.validdata_next_val)[0]
            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:self.batch_size],
                                       self.batch_size)

            for j in range(0, self.batch_size):
                # print(c,j)
                count += 1
                decode_sent = decoded_res[j]
                corrected = correction(decode_sent)
                label = word_batch['source_str'][j]
                #print(label)

                if verbose:
                    cor_wer = wer(label, corrected)
                    dec_wer = wer(label, decode_sent)

                    if(dec_wer < 0.4 or cor_wer < 0.4 or self.force_output):
                        print("\n{}.GroundTruth:{}\n{}.Transcribed:{}\n{}.LMCorrected:{}".format(str(j), label,
                                                                                     str(j), decode_sent,
                                                                                     str(j), corrected))

                    # print("Sample Decoded WER:{}, Corrected LM WER:{}".format(dec_wer, cor_wer))

                originals.append(label)
                results.append(corrected)

        print("########################################################")
        print("Completed Validation Test: WER & LER results")
        rates, mean = wers(originals, results)
        # print("WER rates     :", rates)
        lrates, lmean, norm_lrates, norm_lmean = lers(originals, results)
        # print("LER rates     :", lrates)
        # print("LER norm rates:", norm_lrates)
        # print("########################################################")
        print("Test WER average is   :", mean)
        print("Test LER average is   :", lmean)
        print("Test normalised LER is:", norm_lmean)
        print("########################################################")
        # print("(note both WER and LER use LanguageModel not raw output)")

        self.mean_wer_log.append(mean)
        self.mean_ler_log.append(lmean)
        self.norm_mean_ler_log.append(norm_lmean)

        #delete all values?
        # del originals, results, count, allvalid
        # del word_batch, decoded_res
        # del decode_sent,


    def on_epoch_end(self, epoch, logs=None):
        K.set_learning_phase(0)

        if(self.shuffle_epoch_end):
            print("shuffle_epoch_end")
            self.validdata.genshuffle()


        self.validate_epoch_end(verbose=1)

        if self.save:
            #check to see lowest wer/ler on prev values
            if(len(self.mean_wer_log)>2):
                lastWER = self.mean_wer_log[-1]
                allWER = np.min(self.mean_wer_log[:-1])
                lastLER = self.mean_ler_log[-1]
                allLER = np.min(self.mean_ler_log[:-1])

                if(lastLER < allLER or lastWER < allWER):
                    savedir = "./checkpoints/epoch/LER-WER-best-{}".format(self.runtimestr)
                    print("better ler/wer at:", savedir)
                    if not os.path.isdir(savedir):
                        os.makedirs(savedir)
                    try:
                        save_model(self.model, name=savedir)
                    except Exception as e:
                        print("couldn't save error:", e)

                #early stopping if VAL WER worse 4 times in a row
                if(len(self.mean_wer_log)>5 and self.earlystopping):
                    if(earlyStopCheck(self.mean_wer_log[-5:])):
                        print("EARLY STOPPING")

                        print("Mean WER   :", self.mean_wer_log)
                        print("Mean LER   :", self.mean_ler_log)
                        print("NormMeanLER:", self.norm_mean_ler_log)

                        sys.exit()


        #activate learning phase - incase keras doesn't
        K.set_learning_phase(1)


def decode_batch(test_func, word_batch, batch_size):
    ret = []
    output = test_func([word_batch])[0] #16xTIMEx29 = batch x time x classes
    greedy = True
    merge_chars = True

    for j in range(batch_size):  # 0:batch_size

        if greedy:
            out = output[j]
            best = list(np.argmax(out, axis=1))

            if merge_chars:
                merge = [k for k,g in itertools.groupby(best)]

            else:
                raise ("not implemented no merge")

        else:
            pass
            raise("not implemented beam")

        try:
            outStr = int_to_text_sequence(merge)

        except Exception as e:
            print("Unrecognised character on decode error:", e)
            outStr = "DECODE ERROR:"+str(best)
            raise("DECODE ERROR2")

        ret.append(''.join(outStr))

    return ret

def earlyStopCheck(array):
    last = array[-1]
    rest = array[:-1]
    print(last, " vs ", rest)

    #in other words- the last element is bigger than all 4 of the previous, therefore early stopping required
    if all(i <= last for i in rest):
        return True
    else:
        return False
