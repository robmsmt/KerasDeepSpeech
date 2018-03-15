import os
import argparse
from engine.KerasDeepSpeech import KDS


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))




def main(model_name):

    '''
    There are x simple steps to this program
    '''


    models_path = os.path.join(ROOT_DIR, "models")
    model_meta = {'model_name': model_name,
                'models_path': models_path}

    kds = KDS(model_meta)

    kds.train()

    print(kds.model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='ds1',
                        help='name of model inside models/ folder. Default is ds1')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the learning rate used by the optimiser')
    parser.add_argument('--opt', type=str, default='adam',
                        help='the optimiser to use, default is adam')
    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=2,
                       help='batch_size used to train the model')

    args = parser.parse_args()

    main(args.model_name)
