import numpy as np
import argparse
import pandas as pd
from sklearn import preprocessing

# Preprocesses the different data sets

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='name of the dataset which should be preprocessed', type=str,  default='Wine')

    args = vars(parser.parse_args())

    if args['dataset'] == 'Wine':
        prepWineData()
    elif args['dataset'] == 'Telescope':
        prepTeleData()
    elif args['dataset'] == 'Abalone':
        prepAbaData()
    elif args['dataset'] == 'Heart':
        prepHeartData()
    else:
        raise ValueError("Dataset argument must be one of: Wine, Abalone, Telescope, Heart")

"""
    Preprocess the wine data.
        - Combines the white and the red wine data set to generate binary classification task
        - The data is normalized between 0 and 1
"""
def prepWineData():
    # Load data
    red    = np.genfromtxt(f"./rawData/winequality-red.csv",delimiter=";")[1:]
    white  = np.genfromtxt(f"./rawData/winequality-white.csv",delimiter=";")[1:]

    # Binarize data, 0 is red wine, 1 is white wine
    red = np.concatenate([red,np.zeros([len(red),1])],axis=1)
    white = np.concatenate([white,np.ones([len(white),1])],axis=1)
    data   = np.concatenate([red,white])
    label = data[:,-1]
    label = label.astype(np.float32)

    # Normalize data
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    data = data/np.max(data,axis=0)
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)

    # Save data
    np.save("./data/wine.npy",data)

"""
    Preprocess the wine data.
        - Preprocesses the Telescope data set
        - 1 is defined has 0 for gamma (signal) and 1 for hadron (background)
        - The data is normalized between 0 and 1
"""
def prepTeleData():
    file = open("./rawData/magic04.data", "r").read()
    data = [np.array(file.split("\n")[i].split(",")) for i in range(len(file.split("\n")))]
    data = np.array(data[:-1])

    # Binarize label
    label = data[:,-1]
    label[label=="g"]=0
    label[label=="h"]=1
    label = label.astype(np.float32)

    # Normalize data
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    data = data/np.max(data,axis=0)
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)

    # Save data
    np.save("./data/tele.npy",data)

"""
    Preprocess the wine data.
        - Preprocesses the Telescope data set
        - Binarized by defining class 0 as less equal 9 rings and 1 as more than 9 rings
        - The data is normalized between 0 and 1
"""
def prepAbaData():
    le = preprocessing.LabelEncoder()
    # Load data
    file = open("./rawData/abalone.data", "r").read()
    data = [np.array(file.split("\n")[i].split(",")) for i in range(len(file.split("\n")))]
    data = np.array(data[:-1])
    data[:,0] = le.fit_transform(data[:,0])
    data = data.astype(np.float32)

    # Binarize label
    label = data[:,-1]
    label[label<=9]=0
    label[label>0]=1

    # Normalize data
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    data = data/np.max(data,axis=0)
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)

    # Save data
    np.save("./data/aba.npy",data)

"""
    Preprocess the hearts/diabetes data.
        - Preprocesses the Hearts and Diabetes data sets
        - The hearts data is created by combining the cleveland and hungarian disease data set
        - For hearts: Binarized by defining class 0 as no disease, 1 as disease
        - For diabetes: 0 is tested negative, 1 is tested positive
        - The data is normalized between 0 and 1
"""
def prepHeartData():
    # Load Cleveland data
    file = open("./rawData/processed.cleveland.data", "r").read()
    data = [np.array(file.split("\n")[i].split(",")) for i in range(len(file.split("\n")))]
    data = np.array(data[:-1])
    data[data=="?"]="0.0"
    data = data.astype(np.float32)

    # Binarize labels and normalize
    label = data[:,-1]
    label[label!=0]=1
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    data = data/np.max(data,axis=0)
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)
    data1 = data.astype(np.float32).copy()

    # Load Hungarian data
    file = open("./rawData/processed.hungarian.data", "r").read()
    data = [np.array(file.split("\n")[i].split(",")) for i in range(len(file.split("\n")))]
    data = np.array(data[:-1])
    data[data=="?"]="0.0"
    data = data.astype(np.float32).copy()

    # Binarize labels and normalize
    label = data[:,-1]
    label[label!=0]=1
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    maxs = np.max(data,axis=0)
    maxs[maxs==0] = 1
    data = data/maxs
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)
    data2 = data.astype(np.float32).copy()

    # Combine hearts data sets and save
    data = np.concatenate([data1,data2])
    np.save("./data/heart.npy",data)

    # Load diabetes data
    data = pd.read_csv(f"./rawData/dataset_37_diabetes.arff",delimiter=",")
    data = np.array(data)
    label = data[:,-1]
    label[label=="tested_negative"]="0"
    label[label=="tested_positive"]="1"
    label = label.astype(np.float32)

    # Normalize data and save
    data = data[:,:-1].astype(np.float32)
    data = data-np.min(data,axis=0)
    data = data/np.max(data,axis=0)
    data = np.concatenate([data,np.reshape(label,[label.shape[0],1])],axis=1)
    np.save("./data/dia.npy",data)


if __name__ == '__main__':
    main()