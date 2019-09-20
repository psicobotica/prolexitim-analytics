import pandas as pd
import os

def process_data(data_path, data_file):
    """
    Processes the data extracting only the columns we are interested in:
    :param data_path: path to the directory where the input file is stored.
    :param data_file: file from which the data are extracted.
    :return: a dict with the outputs of the model
    """
    file = os.path.join(data_path, data_file)
    data = pd.read_csv(file, usecols=['Text-EN', 'F1', 'F2', 'F3', 'AlexLabel', 'SubClass'])

    return data