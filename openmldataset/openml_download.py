## Code created by: Lukas Brinkmeyer and Rafael Rego Drumond
## Credits to: Jonas Falkner

### RUN THIS SCRIPT TO DOWNLOAD DATA-SETS FOR  CHAMELEON EXPERIMENTS

from openml_loader import *
import numpy as np
import openml

selected_ids = [6,18 ,23 ,32 ,37 ,50 ,54 ,151,287,307,469,1120 ,1462 ,1464 ,1480 ,1489 ,1497 ,1510 ,1557 ,4538 ,40983,40984,41027]
def download_openmlnew(idnum, root, openml_map_replace = None, verbose=False):

    ID = idnum
    dataset = openml.datasets.get_dataset(ID)
    name = dataset.name
    ddir = os.path.join(root, name)
    if check_exists(ddir):
        x, y = load_np_data(ddir)
        if verbose:
            print(f'Dataset already downloaded and verified at {ddir}.')

    else:
        print(f"Downloading OpenML dataset '{name}'...")
        #dataset = openml.datasets.get_dataset(ID)
        if verbose:
            print("This is dataset '%s', the target feature is '%s'" %
                  (dataset.name, dataset.default_target_attribute))
            print("URL: %s" % dataset.url)
            print(dataset.description)

        x, y = convert_openml_ds(dataset)
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                print( ValueError('NAN values encountered in data!'))
                return None,None,name
        save_np_data(ddir, x, y)
        print(f'Dataset downloaded to {ddir}.')

     

    return x, y, name


validIDs = []
dic = {}
import os
for ii in np.sort(selected_ids):
    i = int(ii)
    x , y, name = download_openmlnew(i, "./Data/selected")
    dic[i] = name
    if x is not None:
        unique, counts = np.unique(y,return_counts=True)
        print (f"id{i}, data shape {x.shape}, num of labels {len(unique)}, {counts} ")
        validIDs.append(i)

        

data_tables = []
for i in validIDs:
    x , y, name = download_openmlnew(i, "./Data/selected")
    if x is not None:
        unique, counts = np.unique(y,return_counts=True)
        print(i,np.mean(x,axis=0))
#         print (f"id{i}, {name}, data shape {x.shape}, num of labels {len(unique)}, {counts} ")
        data_tables.append([i,x.shape[0],x.shape[1],len(unique),counts])

print   (f"{'ID':<6}|{'NAME':<40}|{'SAMPLES':<7}|{'FEAT.s':<6}|{'LABELS':<6}|LABEL COUNT")
tofile = f"{'ID':<6}|{'NAME':<40}|{'SAMPLES':<7}|{'FEAT.s':<6}|{'LABELS':<6}|LABEL COUNT\n"
for dt in data_tables:
    dataset = openml.datasets.get_dataset(dt[0])
    name = dataset.name
    print            (f"{dt[0]:<6}|{name:<40}|{dt[1]:<7}|{dt[2]:<6}|{dt[3]:<6}|{np.mean(dt[4]):.2f}, std:{np.std(dt[4]):.2f}")
    tofile = tofile + f"{dt[0]:<6}|{name:<40}|{dt[1]:<7}|{dt[2]:<6}|{dt[3]:<6}|{np.mean(dt[4]):.2f}, std:{np.std(dt[4]):.2f}, {dt[4]}\n"

with open("SELECT.txt", 'w+') as f:
    f.write(tofile)