 import numpy as np
import random

"""
	Library of all data handling methods
"""

# Combinely shuffles a predictor and label vector
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Shuffles a predictor vector along the feature vector
def shuffleI(vec):
  return vec[:,np.random.permutation(vec.shape[1])]

# Pads the feature column to length "size"
def padCols(vec,size):
    if (size==vec.shape[1]):
        return vec
    return np.concatenate([vec,np.zeros([vec.shape[0],size-vec.shape[1]])],axis=1)

# Pads the instance rows to length "size"
def padRows(vec,size):
    if (size==vec.shape[0]):
        return vec
    return np.concatenate([vec,np.zeros([size-vec.shape[0],vec.shape[1]])],axis=0)

# Loads the hearts and diabetes
def loadHeartData(data_path,perm=[]):

	# Load hearts data
    dataHeart = np.load(f"{data_path}/heart.npy")
    X_train   = dataHeart[:,:-1]
    y_train   = dataHeart[:,-1]

    # Loads diabetes data
    dataDia = np.load(f"{data_path}/dia.npy")
    X_test  = dataDia[:,:-1]
    y_test  = dataDia[:,-1]

	# Shuffle both data sets along instances   
    X_train,y_train = unison_shuffled_copies(X_train,y_train)
    X_test,y_test = unison_shuffled_copies(X_test,y_test)

    # Shuffle both data sets along feature axis for randomized feature splits
    # A permutation vector can be passed
    if len(perm) == 0:
        X_train = shuffleI(X_train)
        X_test = shuffleI(X_test)

    elif len(perm) == len(data[0]) and (np.unique(perm) == list(range(len(data[0])))).all():
        X_train = X_train[:,perm]
        X_test = shuffleI(X_test)

    else:
        raise ValueError("Perm must be [] for random permutation or a valid permutation array")

    return X_train,y_train,X_test,y_test

# Loads standard dataset and creates train/test split with ratio "split"
def loadData(data_path,split = 0.8,featSplits = 0,perm=[]):
	# Load data
    data = np.load(data_path)
    labels = data[:,-1]
    data   = data[:,:-1]
            
    # Shuffle data along instances
    data,labels = unison_shuffled_copies(data,labels)
    
    # Shuffle data along features
    if len(perm) == 0:
       data = shuffleI(data)
    elif len(perm) == len(data[0]) and (np.unique(perm) == list(range(len(data[0])))).all():
       data = data[:,perm]
    else:
       raise ValueError("Perm must be [] for random permutation or a valid permutation array")

    # Split data in Train and Test
    X_train = data[:int(len(data)*split)]
    y_train = labels[:int(len(data)*split)]
    X_test  = data[int(len(data)*split):]
    y_test  = labels[int(len(data)*split):]

    # Split features
    if featSplits != 0:
        X_train = X_train[:,:featSplits]

    return X_train,y_train,X_test,y_test


#Samples a task dataset 
def sample_mini_dataset(number_class,number_shots,min_class,max_class,data,labels,pad,bootstrap=0):
    for class_idx in range(number_class):
        samples = generate_data_batch(number_shots,data,labels,min_class,max_class,pad,bootstrap)
        for sample in range(len(samples[0])):
            yield (samples[0][sample],samples[1][sample])

# Samples data for a random task
def generate_data_batch(batch_size,data,labels,min_class,max_class,pad,bootstrap):
    # Shuffle data
    x,y = unison_shuffled_copies(data,labels)
    x = x[:batch_size]
    y = y[:batch_size]

    # Randomize number of features for task X
    if max_class<=x.shape[1]:
        rand = np.random.randint(min_class,max_class+1)
    else:
        rand = np.random.randint(min_class,x.shape[1])

    # If feature split is used, guarantee features from train and test
    if bootstrap != 0: 
        features1 = np.random.choice(range(bootstrap),np.random.randint(2,6),replace=False) 
        features2 = np.random.choice(range(bootstrap,len(x[0])),np.random.randint(1,3),replace=False) 
        features = np.concatenate([features1,features2])
    else:
        features = np.random.choice(range(len(x[0])),rand,replace=False) 
    out = np.transpose(np.array([x[:,i] for i in features]))

    # If no Chameleon component is used, pad tasks to length K
    if pad != 0:
        out = padCols(out,pad)
    
    return out,y

# Samples minibatches from a mini dataset
def mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if len(samples)==0:
        raise ValueError
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    counter = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return