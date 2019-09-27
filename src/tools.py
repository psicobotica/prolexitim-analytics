import pandas as pd
import os
import argparse
import polyaxon_utils
import types
import logging
from keras.preprocessing import text, sequence

class Configuration(argparse.Namespace):
    """
    Class to store the configuration read with argparse. This file only allow to set once a
    value per key. So the attributes cannot be modified one they are set. If you want to do it you
    have to copy the Configuration and set the new values in the copy method arguments.
    """

    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

    def set(self, name, value):
        if name in self.__dict__ and value != self.__dict__[name]:
            raise Exception('Params cannot be modified')
        else:
            self.__dict__[name] = value

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return NotImplemented
        return vars(self) == vars(other)

    def __ne__(self, other):
        if not isinstance(other, Configuration):
            return NotImplemented
        return not (self == other)

    def __contains__(self, key):
        return key in self.__dict__

    def __setattr__(self, name, value):
        self.set(name, value)

    def __setitem__(self, name, value):
        self.set(name, value)

    def __delattr__(self, name):
        raise Exception('Params cannot be deleted')

    def __delitem__(self, name):
        raise Exception('Params cannot be deleted')

    def copy(self, **kwargs):
        new_params = self.__dict__.copy()
        for key, value in kwargs.iteritems():
            new_params[key] = value
        return Configuration(**new_params)

def get_args():
    """
        This method parses and return arguments passed in
    :return:
    """
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument('-i', '--input_path', help='Path for input files', default=None, type=str, required=True)
    parser.add_argument('-l', '--log_path', help='Path for log files', default=None, type=str, required=False)

    # Network arguments
    parser.add_argument('--factor', type=str, choices=['F1', 'F2', 'F3', 'TAS20'], default='TAS20',
                        help='Factor we are aiming to predict')
    parser.add_argument('-epochs', '--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Minibatch size')
    parser.add_argument('-patience', '--patience', type=int, default=5,
                        help='Number of epochs without change in the accuracy till the training is stopped')

    # File arguments
    parser.add_argument('-train_filename', '--train_filename', type=str, default="train.csv",
                        help='File with dataset for training.')
    parser.add_argument('-test_filename', '--test_filename', type=str, default="test.csv",
                        help='File with dataset for testing.')
    parser.add_argument('-models_path', '--models_path', type=str, default='./data/',
                        help='Directory where the model is stored')
    parser.add_argument('-tokenizer_file', '--tokenizer_file', type=str, default='tokenizer.pk',
                        help='File with tokenizer')
    parser.add_argument('-rt', '--read_tokenizer', action='store_true',
                        help='True when reading the tokenizer from a file instead of building it again.')
    parser.add_argument('-embedding_path', '--embedding_path', type=str, default='./data/',
                        help= 'Directory where the embedding files are stored.')
    parser.add_argument('-embedding_file', '--embedding_file', type=str, default='SBW-vectors-300-min5.txt',
                        help='File with embedding weights.')
    parser.add_argument('-embedding_matrix_file', '--embedding_matrix_file', type=str, default='embedding_matrix.pk',
                        help='File with embedding matrix.')
    parser.add_argument('-weights_file', '--weights_file', type=str, default='weights_base.best.hdf5',
                        help='File with network´s weights.')
    parser.add_argument('--outputs_dir', type=str, default=polyaxon_utils.get_output_path('./outputs'),
                        help='Location where checkpoints and summaries are saved. By default is ./outputs')

    # Embedding arguments:
    parser.add_argument('-max_features', '--max_features', type=int, default=100000,
                        help='Maximum number of words to keep, based on word frequency.')
    parser.add_argument('-embed_size', '--embed_size', type=int, default=300, help='Size of the embedding matrix')


    # Array for all arguments passed to script
    args = parser.parse_args(namespace=Configuration)

    args_vars = vars(args)
    args_vars_list = []
    for k in sorted(args_vars.keys()):
        if not (isinstance(args_vars[k], types.FunctionType) or k.startswith('__')):
            args_vars_list.append('{}: {}'.format(k, args_vars[k]))
    logging.info('Using command args: { %s }', ', '.join(args_vars_list))
    polyaxon_utils.set_params(args_vars)

    return args


def get_tokenizer(args):
    if args.read_tokenizer:
        with open(os.path.join(args.models_path, args.tokenizer_file), 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = text.Tokenizer(num_words=args.max_features, lower=True)
    return tokenizer

def AlextoFloat(x):
    p = 0.0
    if (x == 'Alex'):
        p = 1.0
    return p


def number_sentences(text):
    return len(text.split())


def process_data_classification(file_path):
    """
    Processes the data extracting only the columns we are interested in:
    :param file_path: path to the input  file.
    :return: a pandas dataframe with the data
    """
    file = os.path.join(file_path)
    data = pd.read_csv(file, usecols=['Text-EN','AlexLabel', 'card'])
    data['AlexLabel'] = data.AlexLabel.apply(lambda x: AlextoFloat(x))
    data['number_sentences'] = data['Text-EN'].apply(lambda x: number_sentences(x))

    return data


def process_data_regression(file_path):
    """
    Processes the data extracting only the columns we are interested in:
    :param file_path: path to the file where the input file is stored.
    :return: a pandas dataframe with the data
    """
    data = pd.read_csv(file_path, usecols=['Text-EN','card', 'TAS20', 'F1', 'F2', 'F3'])
    dataDummies = pd.get_dummies(data['card'], prefix='card')
    data = pd.concat([data, dataDummies], axis=1)
    data['number_sentences'] = data['Text-EN'].apply(lambda x: number_sentences(x))

    return data