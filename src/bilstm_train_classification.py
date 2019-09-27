from tools import process_data_classification
from tools import Configuration
from tools import get_args
from tools import get_tokenizer
import logging
import argparse
import os
import pickle
import tensorflow as tf
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


if __name__ == '__main__':

    args = get_args()
    logger.info(f"Reading test file")
    test_file = "/data/serendeepia/prolexitim_alexitimia/20190920_prolexytim_analytics/test.csv"
    test = process_data_classification(test_file)
    X_test = test.drop('AlexLabel', axis=1)
    texts_test = X_test["Text-EN"].str.lower()
    cards_test = X_test.filter(regex="card_.*")  # Using a regular expression cause some cards may not be in test.
    y_test = test['AlexLabel']
    logger.info(f" Using test dataset with {X_test.shape[0]} instances")

    logger.info(f"Reading train file")
    train_file = "/data/serendeepia/prolexitim_alexitimia/20190920_prolexytim_analytics/train.csv"
    train = process_data_classification(train_file)
    X_train = train.drop('AlexLabel', axis=1)
    texts_train = X_train["Text-EN"].str.lower()
    cards_train = X_train.filter(regex="card_.*")  # Using a regular expression cause some cards may not be in train.
    y_train = train['AlexLabel']
    logger.info(f" Using train dataset with {X_train.shape[0]} instances")

    #To do. Adapting test to small sample

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

    # Adding context
    auxiliary_input = Input(shape=(len(cards_train.columns),), name='aux_input')
    x_1 = RepeatVector(args.max_length)(auxiliary_input)
    x = Concatenate()([x, x_1])

    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[text_input, auxiliary_input], outputs=[main_output])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    model.fit([np.array(x_train), cards_train.values], [y_train],
              epochs=args.epochs, batch_size=args.batch_size)

    y_pred = model.predict([x_test, cards_test], batch_size=1024, verbose=1)
    score = roc_auc_score(y_test, y_pred)
    logger.info(f"AUC for test {score}")
    logger.info("Saving model")
    model.save_weights(os.path.join(args.outputs_dir, args.weights_file))
