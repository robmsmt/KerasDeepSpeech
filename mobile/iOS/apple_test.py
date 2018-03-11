import keras
from keras.layers import Input, Dense, GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
import coremltools
import numpy as np


def get_model_1(input_dim, output_dim):
    input_data = Input(name='the_input', shape=(None, input_dim))
    x = input_data
    layercount = 3


    for l in range(layercount):
        x = TimeDistributed(Dense(1024, name='fc_{}'.format(l + 1), activation='relu'))(x)

    x = GRU(1024, return_sequences=True, activation='relu', name='rnn1')(x)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)

    model = Model([input_data], [y_pred])
    return model


input_dim = 26
output_dim = 29


model = get_model_1(input_dim, output_dim)
mlmodel = coremltools.converters.keras.convert(model)


seq_len = 3
cdata = np.random.rand(seq_len, 1, input_dim)
kdata = cdata.transpose((1, 0, 2))
kres = model.predict([kdata])

cres_dict = mlmodel.predict({'input1': cdata})
cres = cres_dict['output1']

print(kres)
print('\n')
print(cres)

# Set model metadata
mlmodel.author = 'rmsmith'
mlmodel.license = 'BSD'
mlmodel.input_description['input1'] = 'Audio input'
mlmodel.output_description['output1'] = 'Audio transcription'

# SAVE CoreML
mlmodel.save('./kds.mlmodel')
