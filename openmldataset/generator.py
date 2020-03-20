## Code created by: Lukas Brinkmeyer and Rafael Rego Drumond

import numpy as np
import os
import random
import pickle


# This class generates few shot classification tasks.

class genFewShot():
    
    def __init__(self, path):
        self.path = path
        self.path_x = os.path.join(path, "features.npy")
        self.path_y = os.path.join(path, "labels.npy"  )
        assert os.path.exists(path)       , f"Error! Directory {path} not found"
        assert os.path.exists(self.path_x), f"Error! Features in {self.path_x} not found"
        assert os.path.exists(self.path_y), f"Error! Labels   in {self.path_y} not found"
        self.path_xe = os.path.join(path, "features_test.npy")
        self.path_ye = os.path.join(path, "labels_test.npy"  )
        self.dataset_x = np.load(os.path.join(path, "features.npy"))
        self.dataset_y = np.load(os.path.join(path, "labels.npy"  ))
        self.test_f  = None
        self.train_f = None
        self.ratio   = None
        self.totalFeatures = self.dataset_x.shape[-1]
        self.totalLabels = len(np.unique(self.dataset_y))
        self.labelList   = np.unique(self.dataset_y)
        self.train_x, self.train_y, self.val_x, self.val_y = None,None,None,None
        self.test_data        = None
        self.test_data_oracle = None

        self.special = False
        if os.path.exists(self.path_xe) and os.path.exists(self.path_ye):
            print("special test features found")
            self.special = True
            self.train_x = self.dataset_x
            self.train_y = self.dataset_y
            self.train_f = range(self.totalFeatures)

            self.val_x  = np.load(os.path.join(path, "features_test.npy"))
            self.val_y  = np.load(os.path.join(path, "labels_test.npy"  ))
            self.test_f = range(self.val_x.shape[-1])

    def setFeatures(self, feats, ratio=None):
        if self.special:
            print("Warning: Not possible to split features between Train and Test with combined datasets")
            return list(self.test_f)

        self.ratio = ratio
        if type(feats) is int:
            self.test_f = np.random.choice(range(self.totalFeatures), feats, False) 
        elif type(feats) is float:
            self.test_f = np.random.choice(range(self.totalFeatures), max(1,int(feats*self.totalFeatures)), False) 
        else:
            assert np.max(np.array(feats)) < self.totalFeatures, "Error, the dataset does not contain such high indexes"
            self.test_f  = feats
        self.train_f = list(set(range(self.totalFeatures)) - set(self.test_f ))
        return list(self.test_f)
    
    def splitTrainVal(self, ratio):
        if self.special:
            print ("Warning: Not possible to split samples between Train and Test with combined datasets")
        else:
            totalSamples = range(len(self.dataset_y))
            labelTotal = self.totalLabels
            train = []
            val   = []
            for y in np.unique(self.dataset_y):
                currentsamples = np.where(self.dataset_y == y)[0]
                val_samples = np.random.choice(currentsamples,int(ratio*float(len(currentsamples))), False)
                for x in val_samples:
                    val.append(x)
                for z in [x for x in currentsamples if x not in val_samples]:
                    train.append(z)
            self.train_x, self.train_y = self.dataset_x[train], self.dataset_y[train]
            self.val_x  , self.val_y   = self.dataset_x[val]  , self.dataset_y[val]
            
    def featSplit(self, ammount, test=False):
        feats = []
        if self.special and test:
                feats = feats + list(np.random.choice(self.test_f, ammount, False))
        else:
            if test and self.ratio is not None and self.ratio > 0:
                t_ammount = min(int(np.maximum(1., self.ratio * float(ammount))),len(self.test_f))
                ammmount  = int(np.maximum(1., float(ammount) - float(t_ammount)))
                feats = feats + list(np.random.choice(self.test_f, t_ammount, False))
            feats = feats + list(np.random.choice(self.train_f, ammount, False))
        return feats
        
    def zeroColumns(self,vec,feats):
        dummy  = vec
        totalF = range(self.totalFeatures)
        for i in totalF:
            if i not in feats:
                dummy[:,i] = 0.0
        return dummy
    
    def generate(self,
                 num_labels,
                 min_f,
                 max_f,
                 train_shots,
                 val_shots,
                 meta_batch,
                 test = False,
                 sort = True,
                 oracle = False,
                 double = False):

        if type(min_f)==float:
            min_f = max(1,int(min_f*len(self.train_f)))
        if type(max_f)==float:
            max_f = max(1,int(max_f*len(self.train_f)))

#         count = -1
        assert self.val_x is not None  , "please run splitTrainVal(ratio) first! val_x"
        assert self.val_y is not None  , "please run splitTrainVal(ratio) first! val_y"
        assert self.train_x is not None, "please run splitTrainVal(ratio) first! train_x"
        assert self.train_y is not None, "please run splitTrainVal(ratio) first! train_y"
        assert min_f <= max_f                  , f"Error minimum features should be smaller than maximum features (DUH!)"
        assert num_labels <= self.totalLabels  , f"Error: you have given {num_labels} labels, but your data only has {self.totalLabels}"
        assert max_f      <= self.totalFeatures, f"Error: you have given {max_f} max features, but your data only has {self.totalFeatures}"
        assert min_f       > 0                     , f"Error: you have given {min_f} min features, please select an integer number higher than zero"
        assert train_shots > 0                     , f"Error: you have given {train_shots} t-shots, please select an integer number higher than zero"
        assert val_shots   > 0                     , f"Error: you have given {val_shots} v-shots, please select an integer number higher than zero"
       # assert (oracle != self.special) or (not oracle and not special)              , f"You cannot use oracle tasks when using combined data-sets!"

        labelvect = np.zeros((train_shots * num_labels,num_labels))
        labelvecv = np.zeros((val_shots   * num_labels,num_labels))
        ordert = np.array(range(train_shots * num_labels))
        orderv = np.array(range(val_shots   * num_labels))
        for i in range(num_labels):
            labelvect[i*train_shots:i*train_shots + train_shots,i] = 1.0
            labelvecv[i*val_shots  :i*val_shots   + val_shots  ,i] = 1.0
        if test:
            samples_x = self.val_x
            samples_y = self.val_y
        else:
            samples_x = self.train_x
            samples_y = self.train_y
        while True:
#             count += 1
            mb_tx, mb_ty, mb_vx, mb_vy = [],[],[],[]
            if oracle:
                mbo_tx, mbo_ty, mbo_vx, mbo_vy = [],[],[],[]
            ammount_features = np.random.randint(min_f, max_f)
            for _ in range(meta_batch):
                labels = np.random.choice(self.labelList , num_labels, False) #random.choices(self.labelList, k=num_labels)
                if sort:
                    labels = np.sort(labels)
                currentlabel = -1
                tr  = []
                val = []
                for l in labels:
                    currentlabel += 1
                    idxs = np.random.choice(np.where(samples_y == l)[0], train_shots+val_shots, False)
                    train_idx = idxs[:train_shots]
                    val_idx   = idxs[train_shots:]
                    tr.append (samples_x[train_idx])
                    val.append(samples_x[val_idx]  )
                np.random.shuffle(ordert)
                np.random.shuffle(orderv)
                feats = self.featSplit(ammount_features, test)
                a,b,c,d = np.concatenate(tr)[ordert].astype(np.float32), labelvect[ordert], np.concatenate(val)[orderv].astype(np.float32), labelvecv[orderv]
                mb_tx.append(a[:,feats])
                mb_ty.append(b)
                mb_vx.append(c[:,feats])
                mb_vy.append(d)
                
                if oracle:
                    mbo_tx.append(self.zeroColumns(a,feats))
                    mbo_ty.append(b)
                    mbo_vx.append(self.zeroColumns(c,feats))
                    mbo_vy.append(d)
                
            if oracle:
                if double:
                    yield (np.array(mb_tx), np.array(mb_ty), np.array(mb_vx), np.array(mb_vy)), (np.array(mbo_tx), np.array(mbo_ty), np.array(mbo_vx), np.array(mbo_vy))
                else:
                    yield np.array(mbo_tx), np.array(mbo_ty), np.array(mbo_vx), np.array(mbo_vy)
            else:
                yield np.array(mb_tx), np.array(mb_ty), np.array(mb_vx), np.array(mb_vy)
            
    
    def generatorFixed(self, data):
        ammount_of_mbs = len(data)
        idxs = np.array(range(ammount_of_mbs))
        np.random.shuffle(idxs)
        while True:
            for idx in idxs:
#                 xx,xy,vx,vy = data[idx]
                yield data[idx]
            np.random.shuffle(idxs)

    def generateDataset(self,
                        num_labels,
                        min_f,
                        max_f,
                        train_shots,
                        val_shots,
                        meta_batch,
                        number_of_mbs=10000,
                        test=False,
                        newPath=None,
                        save=False,
                        oracle = False):
        #assert (oracle != self.special) or (not oracle and not special), "You cannot use oracle tasks when using combined data-sets!"
        if type(min_f)==float:
            min_f = max(1,int(min_f*len(self.train_f)))
        if type(max_f)==float:
            max_f = max(1,int(max_f*len(self.train_f)))


        double = False
        if oracle and test:
            double = True
        g = self.generate(num_labels, min_f, max_f, train_shots, val_shots, meta_batch, test=test, oracle=oracle, double=double)
        tasks   = []
        tasks_o = []
        for i in range(number_of_mbs):
            if double:
                nor, orc = next(g)
                tasks.append(nor)
                tasks_o.append(orc)
            else:
                tasks.append(next(g))
        
        currentPath = self.path
        if newPath is not None:
            currentPath = newPath
            os.system("mkdir -p "+newPath)
        
        path = os.path.join(currentPath, "tasksTrain.pkl")
        if test:
            path  = os.path.join(currentPath, "tasksTest.pkl" )
            patho = os.path.join(currentPath, "tasksTestOracle.pkl" )
            self.test_data = tasks
            if double:
                self.test_data_oracle = tasks_o

        if save:
            with open(path, 'wb') as f:
                pickle.dump(tasks, f)
            if double:
                with open(patho, 'wb') as f:
                    pickle.dump(tasks_o, f)
#         np.save(path, tasks)
        if double:
            return tasks, tasks_o
        return tasks
            

class preLoadedGen():
    def __init__(self, path, train=True, test=True):
        self.path_tr = os.path.join(path, "tasksTrain.pkl")
        self.path_ts = os.path.join(path, "tasksTest.pkl" )
        assert os.path.exists(path)       , f"Error! Directory {path} not found"
        assert os.path.exists(self.path_tr), f"Error! Features in {self.path_tr} not found"
        assert os.path.exists(self.path_ts), f"Error! Features in {self.path_ts} not found"
        # self.tasks_tr = np.load(self.path_tr)
        # self.tasks_ts = np.load(self.path_ts)
        if train:
            with open(self.path_tr, 'rb') as f:
                self.tasks_tr = pickle.load(f)
        if test:
            with open(self.path_ts, 'rb') as f:
                self.tasks_ts = pickle.load(f)
        
    def generator(self, test=False):
        if test:
            data = self.tasks_ts
        else:
            data = self.tasks_tr
        ammount_of_mbs = len(data)
        idxs = np.array(range(ammount_of_mbs))
        np.random.shuffle(idxs)
        while True:
            for idx in idxs:
#                 xx,xy,vx,vy = data[idx]
                yield data[idx]
            np.random.shuffle(idxs)
