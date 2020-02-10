### Code modified by: Lukas Brinkmeyer and Rafael Rego Drumond
### Credits to original coding by: Jonas Falkner
import os
import warnings
import logging
import numpy as np
import openml
import pandas as pd

logger = logging.getLogger(__name__)



def check_exists(directory):
    return (os.path.exists(os.path.join(directory, 'features.npy')) and
            os.path.exists(os.path.join(directory, 'labels.npy')))


def save_np_data(directory, x, y):
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, 'features.npy'), x)
    np.save(os.path.join(directory, 'labels.npy'), y)


def load_np_data(directory, split_label=''):
    x = np.load(os.path.join(directory, f'{split_label}features.npy'))
    y = np.load(os.path.join(directory, f'{split_label}labels.npy'))
    return x, y


def cat_np_data(directory):
    """Loads and concatenates data in numpy array format

    Args:
        directory (str): path to directory

    Returns:
        train- and test data concatenated to one data set
    """

    x_train, y_train = load_np_data(directory, split_label='train_')
    x_test, y_test = load_np_data(directory, split_label='test_')

    X = np.concatenate([x_train, x_test], axis=0)
    Y = np.concatenate([y_train, y_test], axis=0)

    return X, Y


def convert_openml_ds(dataset):
    """Converts a openml dataset to numpy arrays and integer labels"""
    x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # dummy code all categorical variables
    x = pd.get_dummies(x)

    # convert to numpy array
    x = x.to_numpy()

    # convert y to integer labels
    y_uniques = np.unique(y)
    n_classes = len(y_uniques)
    int_labels = np.arange(n_classes)
    replacer = dict(zip(y_uniques, int_labels))
    y_list = y.tolist()
    y = np.array(list(map(replacer.get, y_list, y_list)))

    return x, y


def download_openml(name, root, verbose=False):

    # dataset selection criteria:
    # - OpenML-CC18 (tag: study_99)
    # - more than 3000 instances
    # - more than 50 features
    # - MinorityClassPercentage > 3% (multi-label) > 10% (binary)

    openml_map = {                  # n_features, n_classes
        'har': 1478,                # 561, 6
        #'PhishingWebsites': 4534,
        'nomao': 1486,              # 174, 2
        #'connect4': 40668,
        #'numerai':  23517,
        'internetAds': 40978,       # 3113, 2
        'dna': 40670,               # 360, 3
        #'semeion': 1501,
        'splice': 46,               # 287, 3
        'isolet': 300,              # 617, 26
        #'ozone': 1487,
        'spambase': 44,             # 57, 2
        'theorem': 1475,            # 51, 6
        'bioresponse': 4134,        # 1776, 2
        'optdigits': 28,            # 64, 10
    }
    ID = openml_map[name]

    ddir = os.path.join(root, name)

    if check_exists(ddir):
        x, y = load_np_data(ddir)
        if verbose:
            print(f'Dataset already downloaded and verified at {ddir}.')

    else:
        print(f"Downloading OpenML dataset '{name}'...")
        dataset = openml.datasets.get_dataset(ID)
        if verbose:
            print("This is dataset '%s', the target feature is '%s'" %
                  (dataset.name, dataset.default_target_attribute))
            print("URL: %s" % dataset.url)
            print(dataset.description)

        x, y = convert_openml_ds(dataset)
        save_np_data(ddir, x, y)
        print(f'Dataset downloaded to {ddir}.')

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError('NAN values encountered in data!')

    return x, y


