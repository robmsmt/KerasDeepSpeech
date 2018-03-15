

class ModelSchema(object):
    '''
         ModelSchema class

    '''

    def __init__(self, model_meta):




        self.models_path = model_meta['models_path']
        self.model_name = model_meta['model_name']

        import importlib
        s = "models." + self.model_name
        imported_model = importlib.import_module(s)

        self.schema = imported_model.model_schema
        self.settings = imported_model.model_settings

        self.model = self.schema.model()


        print(self.schema)

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
        self.model = ModelSchema(model_meta)

    def train(self):
        pass
