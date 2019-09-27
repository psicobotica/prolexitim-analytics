from tools import process_data_regression
import logging
import argparse
import polyaxon_utils

import os
import pickle

import argparse
import tensorflow as tf
import types


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, concatenate, SpatialDropout1D, Reshape, RepeatVector
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




logger = logging.getLogger(__name__)


def setup_script(filename_log=None):

    # config log
    fmt = '%(asctime)s : %(levelname)s : %(message)s'
    datefmt = '%m%d %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
    )
    if filename_log:
        build_log = logging.FileHandler(filename=filename_log, mode='w')
        build_log.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        build_log.setFormatter(formatter)
        logging.getLogger('').addHandler(build_log)

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
    parser.add_argument('-l', '--log_path', help='Path for log files', default=None, type=str, required=False)
    parser.add_argument('-i', '--input_path', help='Path for input files', default=None, type=str, required=True)

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
    parser.add_argument('-max_len', '--max_length', type=int, default=150, help='Maximum length of the sequences.')
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


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


if __name__ == '__main__':

    args = get_args()
    logger.info(f"Reading test file")
    test = process_data_regression(args.input_path, args.test_filename)
    X_test = test.drop(args.factor, axis=1)
    texts_test = X_test["Text-EN"].str.lower()
    cards_test = X_test.filter(regex="card_.*")  # Using a regular expression cause some cards may not be in test.
    y_test = test[args.factor]
    logger.info(f" Using test dataset with {X_test.shape[0]} instances")

    logger.info(f"Reading train file")
    train = process_data_regression(args.input_path, args.train_filename)
    X_train = train.drop(args.factor, axis=1)
    texts_train = X_train["Text-EN"].str.lower()
    cards_train = X_train.filter(regex="card_.*")  # Using a regular expression cause some cards may not be in train.
    y_train = train[args.factor]
    y_train_norm = (y_train - y_train.mean())/y_train.std()
    y_test_norm = (y_test - y_train.mean())/y_train.std()
    logger.info(f" Using train dataset with {X_train.shape[0]} instances")

    tok = get_tokenizer(args)
    tok.fit_on_texts(list(texts_train) + list(texts_test))
    texts_train = tok.texts_to_sequences(texts_train)
    texts_test = tok.texts_to_sequences(texts_test)
    x_train = sequence.pad_sequences(texts_train, maxlen=args.max_length)
    x_test = sequence.pad_sequences(texts_test, maxlen=args.max_length)

    embeddings_index = {}
    with open(os.path.join(args.embedding_path, args.embedding_file), encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients

    word_index = tok.word_index
    # prepare embedding matrix
    num_words = min(args.max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, args.embed_size))
    for word, i in word_index.items():
        if i >= args.max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    text_input = Input(shape=(args.max_length,), name='text_input')
    x = Embedding(args.max_features, args.embed_size, weights=[embedding_matrix], trainable=False)(text_input)
    x = SpatialDropout1D(0.2)(x)

    # Adding context
    auxiliary_input = Input(shape=(len(cards_train.columns),), name='aux_input')
    x_1 = RepeatVector(args.max_length)(auxiliary_input)
    x = keras.layers.concatenate([x, x_1])

    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    main_output = Dense(1, name='main_output')(x)

    model = Model(inputs=[text_input, auxiliary_input], outputs=[main_output])
    model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    model.fit([np.array(x_train), cards_train.values], [y_train_norm],
              epochs=args.epochs, batch_size=args.batch_size)

    # Little issue for dummies:
    # val_mse, val_mae = model.evaluate([np.array(x_test), cards_test.values], [y_test_norm])
    # print(val_mse)
