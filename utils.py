from char_map import char_map, index_map


from pympler import muppy, summary, tracker, classtracker
from pympler.garbagegraph import GarbageGraph, start_debug_garbage
from pympler.web import start_profiler, start_in_background
import types

import resource
import tensorflow as tf
import keras
from keras.models import model_from_json, load_model
import keras.backend as K

import inspect
import re
import sys

import h5py
import yaml

from model import clipped_relu, selu

# these text/int characters are modified
# from the DS2 github.com/baidu-research/ba-dls-deepspeech

def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_to_text_sequence(seq):
    """ Use a index map and convert int to a text sequence
        >>> from utils import int_to_text_sequence
        >>> a = [2,22,10,11,21,2,13,11,6,1,21,2,8,20,17]
        >>> b = int_to_text_sequence(a)
    """
    text_sequence = []
    for c in seq:
        if c == 28: #ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence




def save_trimmed_model(model, name):

    jsonfilename = str(name) + ".json"
    weightsfilename = str(name) + ".h5"

    # # serialize model to JSON
    with open(jsonfilename, "w") as json_file:
        json_file.write(model.to_json())

    # # serialize weights to HDF5
    model.save_weights(weightsfilename)

    return

def save_model(model, name):

    if name:
        jsonfilename = str(name) + "/model.json"
        weightsfilename = str(name) + "/model.h5"

        # # serialize model to JSON
        with open(jsonfilename, "w") as json_file:
            json_file.write(model.to_json())

        print("Saving model at:", jsonfilename, weightsfilename)
        model.save_weights(weightsfilename)

        #save model as combined in single file - contrains arch/weights/config/state
        model.save(str(name)+"/cmodel.h5")

    return

def load_model_checkpoint(path, summary=True):

    #this is a terrible hack
    from keras.utils.generic_utils import get_custom_objects
    # get_custom_objects().update({"tf": tf})
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    # get_custom_objects().update({"TF_NewStatus": None})

    jsonfilename = path+".json"
    weightsfilename = path+".h5"

    json_file = open(jsonfilename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    K.set_learning_phase(1)
    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights(weightsfilename)
    # loaded_model = load_model(path, custom_objects=custom_objects)


    if(summary):
        loaded_model.summary()

    return loaded_model

def load_cmodel_checkpoint(path, summary=True):

    #this is a terrible hack
    from keras.utils.generic_utils import get_custom_objects
    # get_custom_objects().update({"tf": tf})
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    # get_custom_objects().update({"TF_NewStatus": None})

    cfilename = path+".h5"

    K.set_learning_phase(1)
    loaded_model = load_model(cfilename)


    if(summary):
        loaded_model.summary()

    return loaded_model


memlist = []
class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        web_browser_debug = True
        print(x)

        if x > 40000:
            if web_browser_debug:
                if epoch==0:
                    start_in_background()
                    tr = tracker.SummaryTracker()
                    tr.print_diff()
            else:
                global memlist
                all_objects = muppy.get_objects(include_frames=True)
                # print(len(all_objects))
                sum1 = summary.summarize(all_objects)
                memlist.append(sum1)
                summary.print_(sum1)
                if len(memlist) > 1:
                    # compare with last - prints the difference per epoch
                    diff = summary.get_diff(memlist[-2], memlist[-1])
                    summary.print_(diff)
                my_types = muppy.filter(all_objects, Type=types.ClassType)

                for t in my_types:
                    print(t)


    #########################################################

