import os
import numpy as np
import argparse

"""
    Main.
    Downloads the datasets from UCI/OpenML
"""

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='name of the dataset which should be preprocessed', type=str,  default='Wine')

    args = vars(parser.parse_args())

    download(args["dataset"])

# Downloads the defined data sets, must be one of {Abalone, Wine, Telescope, Heart}
def download(dataset):
    if dataset == "Abalone":
        os.system("wget -P ./rawData https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
    elif dataset == "Wine": 
        os.system("wget -P ./rawData http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
        os.system("wget -P ./rawData http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
    elif dataset == "Telescope":
        os.system("wget -P ./rawData https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data")
    elif dataset == "Telescope":
        os.system("wget -P ./rawData https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data")
        os.system("wget -P ./rawData https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data")
    elif dataset == "Heart":
        os.system("wget -P ./rawData http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
        os.system("wget -P ./rawData http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data")
        os.system("wget -P ./rawData https://www.openml.org/data/get_csv/37/dataset_37_diabetes.arff")
    else:
        raise ValueError("Dataset argument must be one of: Wine, Abalone, Telescope, Heart")
if __name__ == '__main__':
    main()