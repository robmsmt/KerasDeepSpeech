from __future__ import unicode_literals

import argparse

from model import *
from utils import *

import keras
import coremltools
import os

#####################################################

#### MAIN



def main(checkpointpath):
    K.set_learning_phase(0)
    ## hack required for clipped relu
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})


    ## 3. Load existing or error
    if args.loadcheckpointpath:
        # load existing
        print("Loading model")

        cp = args.loadcheckpointpath
        assert(os.path.isdir(cp))

        model_path = os.path.join(cp, "model")
        # assert(os.path.isfile(model_path))

        loaded_model = load_model_checkpoint(model_path)
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        loaded_model.compile(optimizer=opt, loss=ctc)

        print("Model loaded")

    else:
        # new model
        raise("You need to load an existing trained model")

    ## Try to convert assume newly trained and will will fail with CTC lambda
    print("Try convert with CoreML ")
    try:
        coreml_model = coremltools.converters.keras.convert(loaded_model)

    except Exception as e:
        print("Conversion failed - trying to rebuild without lambda")
        print(e)


        ## Rebuild function without CTC lambda and transfer weights
        #todo, get these values automatically. They can be taken from args with meta

        if(args.model_arch == 0):
            model = build_ds0_no_ctc_and_xfer_weights(loaded_model=loaded_model,
                                                                 input_dim=26,
                                                                 fc_size=512,
                                                                 rnn_size=512,
                                                                 dropout=[0.0, 0.0, 0.0],
                                                                 output_dim=29)
        elif (args.model_arch == 1):
            model = build_ds0_no_ctc_and_xfer_weights(loaded_model=loaded_model,
                                                                 input_dim=26,
                                                                 fc_size=512,
                                                                 rnn_size=512,
                                                                 output_dim=29)
        elif (args.model_arch == 5):
            model = build_ds5_no_ctc_and_xfer_weights(loaded_model=loaded_model,
                                                                 input_dim=161,
                                                                 fc_size=512,
                                                                 rnn_size=512,
                                                                 output_dim=29)

        elif(args.model_arch == 6):
            model = build_const_no_ctc_and_xfer_weights(loaded_model=loaded_model,
                                                                 input_dim=26,
                                                                 fc_size=512,
                                                                 rnn_size=512,
                                                                 output_dim=29)
        else:
            raise "no model provided"

        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        print(model.summary(line_length=80))

        print("Retry converting new model")
        coreml_model = coremltools.converters.keras.convert(model)



    # Set model metadata
    coreml_model.author = 'Rob Smith'
    coreml_model.license = 'BSD'
    coreml_model.short_description = 'Performs keras ds '
    coreml_model.input_description['input1'] = 'Audio input'
    coreml_model.output_description['output1'] = 'Audio transcription'

    # SAVE CoreML
    coreml_model.save('./iOS/coreml/kds.mlmodel')

    ##Export the trimmed model (without CTC) to test that it works on python
    save_trimmed_model(model, name='./checkpoints/trimmed/TRIMMED_ds_model')


    ########## TF

    print("export graph to TF")

    sess = K.get_session()
    output_fld = "./iOS/tensorflow_graph/"
    full = args.loadcheckpointpath+str('/model.h5')
    model.load_weights(full)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.save(sess, './iOS/model.ckpt')

    # tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, name='model.pb', as_text=False)

    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)


    # num_output = 1
    # prefix_output_node_names_of_final_network = 'output_node'
    # pred = [None] * num_output
    # pred_node_names = [None] * num_output
    # for i in range(num_output):
    #     pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
    #     pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
    # print('output nodes names are: ', pred_node_names)

    print("Completed")
    K.clear_session()


if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    ##defaults to the finished checkpoint
    parser.add_argument('--loadcheckpointpath', type=str,
                        default='./checkpoints/epoch/LER-WER-best-DS5_2017-08-29_11-37',

                        help='If value set, load the checkpoint json '
                             'weights assumed as same name '
                             ' e.g. --loadcheckpointpath ./checkpoints/'
                             'TRIMMED_ds_ctc_model ')

    parser.add_argument('--model_arch', type=int, default=5,
                       help='choose between model_arch versions (when training not loading) '
                            '--model_arch=1 uses DS1 fully connected layers with LSTM'
                            '--model_arch=2 uses DS2 fully connected with GRU'
                            '--model_arch=3 is custom model'
                        )

    args = parser.parse_args()

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args)

