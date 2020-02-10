# Chameleon V2

## Created by: Lukas Brinkmeyer and Rafael Rego Drumond

## CREDITS
   This code is built on top of Reptile's original implenentation from:
   
    
        Alex Nichol, Joshua Achiam, John Schulman
        Website: https://openai.com/blog/reptile/
        Git:     https://github.com/openai/supervised-reptile
        Paper:   https://arxiv.org/abs/1803.02999
        BIBTEX:
           @article{nichol2018first,
                 title={On first-order meta-learning algorithms},
                 author={Nichol, Alex and Achiam, Joshua and Schulman, John},
                 journal={CoRR, abs/1803.02999},
                 volume={2},
                 year={2018}
           }
     

   If you use our code, please reference the paper above and our [paper](https://arxiv.org/abs/1909.13576):
   
     
     Chameleon: Learning Model Initializations Across Tasks With Different Schemas
     Lukas Brinkmeyer and Rafael Rego Drumond
     BIBTEX:
        @misc{brinkmeyer2019chameleon,
          title={Chameleon: Learning Model Initializations Across Tasks With Different Schemas},
          author={Lukas Brinkmeyer and Rafael Rego Drumond and Randolf Scholz and Josif Grabocka and Lars Schmidt-Thieme},
          year={2019},
          eprint={1909.13576},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
        }
     


## DOWNLOADING DATASETS:

   Most of the used datasets are in OpenML
   Run the script ```openmldataset/openml_download.py``` to download the datasets used in the paper.
   In each folder you will have ```features.npy``` and ```labels.npy```, if you want combined experiments copy from the other folders as ```features_test.npy``` and ```labels_test.npy``` to be used as meta-test-set

## RECOMMENDED PACKAGES:

   You can check our recommended packages in the file ```recommended.txt```

## ARGUMENTS:

   Run the ```run.py``` to run the code, you can use the following arguments.
   Resulting learning curves will be saved in the ```results``` folder
```
  --seed SEED           random seed (default: 0)
  --checkpoint CHECKPOINT
                        checkpoint directory (default: model_checkpoint)
  --save_path SAVE_PATH
                        checkpoint directory (default: exp_1581008235)
  --num_jobs NUM_JOBS   Number of jobs to run in parallel (default: 5)
  --inner_batch INNER_BATCH
                        inner batch size (default: 30)
  --inner_iters INNER_ITERS
                        inner iterations (default: 10)
  --learning_rate LEARNING_RATE
                        Adam step size (default: 0.0001)
  --meta_step META_STEP
                        meta-training step size (default: 0.01)
  --meta_batch META_BATCH
                        meta-training batch size (default: 1)
  --meta_iters META_ITERS
                        meta-training iterations (default: 15001)
  --min_feats MIN_FEATS
                        Min number of features (default: 4)
  --max_feats MAX_FEATS
                        Max number of features (default: 8)
  --freeze FREEZE       whether a permuting network is added (default: False)
  --conv_layers CONV_LAYERS
                        Number and size of conv layers (default: [8,16,14])
  --base_layers BASE_LAYERS
                        Number and size of base layers (default: [64,64])
  --perm_epochs PERM_EPOCHS
                        training epochs for permuter (default: 501)
  --perm_lr PERM_LR     permuter learning rate (default: 0.0001)
  --num_test_features NUM_TEST_FEATURES
                        Ratio of feature split for train test (default: 0)
  --test_feat_ratio TEST_FEAT_RATIO
                        Ratio of feature split for train test (default: 0.0)
  --name NAME           name add-on (default: Model_config-1581008235)
  --dataset DATASET     data set to evaluate on (default: codrna)
  --data_dir DATA_DIR   Path to datasets (default: ./Data/selected)
  --config CONFIG       json config file (default: None)
```

--inner_iters 5 --meta_iters 5 --perm_epochs 5