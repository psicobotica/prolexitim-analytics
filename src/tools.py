import pandas as pd
import os

def AlextoFloat(x):
    p = 0.0
    if (x == 'Alex'):
        p = 1.0
    return p


def process_data_classification(data_path, data_file):
    """
    Processes the data extracting only the columns we are interested in:
    :param data_path: path to the directory where the input file is stored.
    :param data_file: file from which the data are extracted.
    :return: a dict with the outputs of the model
    """
    file = os.path.join(data_path, data_file)
    data = pd.read_csv(file, usecols=['Text-EN','AlexLabel', 'card'])
    data['AlexLabel'] = data.AlexLabel.apply(lambda x: AlextoFloat(x))

    return data


def process_data_regression(data_path, data_file):
    """
    Processes the data extracting only the columns we are interested in:
    :param data_path: path to the directory where the input file is stored.
    :param data_file: file from which the data are extracted.
    :return: a dict with the outputs of the model
    """
    file = os.path.join(data_path, data_file)
    data = pd.read_csv(file, usecols=['Text-EN','card', 'TAS20', 'F1', 'F2', 'F3'])
    dataDummies = pd.get_dummies(data['card'], prefix='card')
    data = pd.concat([data, dataDummies], axis=1)

    return data