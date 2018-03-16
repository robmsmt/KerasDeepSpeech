import importlib
from keras import backend as K
from keras.optimizers import Adam, Nadam, SGD

class ModelSchema(object):
    '''
         ModelSchema class

    '''

    def __init__(self, model_meta):

        self.models_path = model_meta['models_path']
        self.model_name = model_meta['model_name']

        import_model_path = "models." + self.model_name
        imported_model = importlib.import_module(import_model_path)

        self.schema = imported_model.model_schema
        self.settings = imported_model.model_settings

        self.model = self.schema.model()

        self.y_pred = self.model.get_layer('ctc').input[0]
        self.input_data = self.model.get_layer('the_input').input
        self.report = K.function([self.input_data, K.learning_phase()], [self.y_pred])


class KDS(object):
    '''
     KerasDeepSpeech class

     Initialise order:
      1. Loads model (either new from schema or checkpoint)

      Getters & Setters
      2. (optional) Loads Train/Validation CSV data
      3. (optional) Sets post processing language model

     Available methods to KDSModel:
      - train -IN-> requires data + model (opt params) loaded
             -OUT-> produces DF transcript at end & WER report
                    logs & tensorboard info

      - test_batch -IN-> requires data + model (opt params) loaded
                  -OUT-> produces DF transcript at end & WER report

      - live_test -IN-> requires model
                  -OUT-> transcript

    '''


    def __init__(self, model_meta):
        self.schema = ModelSchema(model_meta)
        self.data = None
        self.lm = None


    def data_init(self, args):
        self.data = args




    def train(self, args):



        if (args.opt.lower() == 'sgd'):
            opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        elif (args.opt.lower() == 'adam'):
            opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
        elif (args.opt.lower() == 'nadam'):
            opt = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
        else:
            raise "optimiser not recognised"

        self.schema.model.compile(optimizer=opt, loss=ctc)

        self.schema.model.fit_generator(generator=traindata.next_batch(),
                            steps_per_epoch=args.train_steps,
                            epochs=args.epochs,
                            #callbacks=cb_list,
                            validation_data=validdata.next_batch(),
                            validation_steps=args.valid_steps,
                            initial_epoch=0,
                            verbose=1,
                            class_weight=None,
                            max_q_size=10,
                            workers=1,
                            pickle_safe=False
                            )
