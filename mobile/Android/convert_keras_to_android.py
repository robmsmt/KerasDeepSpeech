# this file is based on https://github.com/amir-abdi/keras_to_tensorflow/

import os
import os.path as osp

from keras import backend as K
from keras.models import load_model

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from utils import load_model_checkpoint, save_model

# SET PARAMS

input_fld = './checkpoints/trimmed/'
model_file = 'TRIMMED_ds_model' #dont use extension e.g .json


num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'constant_graph_weights.pb'

## INIT

output_fld = "./Android/" + 'tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)


## LOAD KERAS MODEL AND RENAME OUTPUT

K.set_learning_phase(0)
net_model = load_model_checkpoint(input_fld+model_file)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)


# WHY [optional] write graph definition in asci??

sess = K.get_session()
if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

# convert variables to constants and save

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))


##### safety check graph
pbfile = osp.join(output_fld, output_graph_name)

g = tf.GraphDef()
g.ParseFromString(open(pbfile, "rb").read())
print([n for n in g.node if n.name.find("input") != -1]) # same for output or any other node you want to make sure is ok
print([n for n in g.node if n.name.find("out") != -1]) # same for output or any other node you want to make sure is ok

#the_input
#output_node0


##weird ops might not be defualt
ops = set([n.op for n in g.node])
print(ops)



