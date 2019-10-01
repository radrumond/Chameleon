Created by: Lukas Brinkmeyer and Rafael Rego Drumond

## CREDITS
   This code is built on top of Reptile's original implenentation from:
   
    ```
        Alex Nichol, Joshua Achiam, John Schulman
        Website: https://openai.com/blog/reptile/
        Git:     https://github.com/openai/supervised-reptile
        Paper:   https://arxiv.org/abs/1803.02999
        @article{nichol2018first,
              title={On first-order meta-learning algorithms},
              author={Nichol, Alex and Achiam, Joshua and Schulman, John},
              journal={CoRR, abs/1803.02999},
              volume={2},
              year={2018}
        }
     ```

   If you use our code, please reference the paper above and our [paper](https://arxiv.org/abs/1909.13576):
   
     ```
     Chameleon: Learning Model Initializations Across Tasks With Different Schemas
     Lukas Brinkmeyer and Rafael Rego Drumond
     @misc{brinkmeyer2019chameleon,
       title={Chameleon: Learning Model Initializations Across Tasks With Different Schemas},
       author={Lukas Brinkmeyer and Rafael Rego Drumond and Randolf Scholz and Josif Grabocka and Lars Schmidt-Thieme},
       year={2019},
       eprint={1909.13576},
       archivePrefix={arXiv},
       primaryClass={cs.LG}
     }
     ```
 
 ## ENVIRONMENT
   The required python 3 libraries for running this file are present in the reccommended.txt file
    
   From the miniconda base instalation you only need to install:
        tensorflow=1.12.0
        scikit-learn-0.20.2
   Make sure numpy is also apropriately installed.
    
   Miniconda 3.7 link:
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

## RUNNING
   Hyper-parameters can be configured in code/args.py
   In order to run the experiments you must:
    
   - Download the data-set with the command below (requires linux due to wget command, in other os's you might need to download the links manually):
        ```python downloader.py --dataset Wine```
        
   - Prep the downloaded data:
        ```python prepData.py --dataset Wine```
        
   - Run the experiment:
        ```python run_perm.py  --checkpoint "./Run" --permuter True --feature_split 8 --dataset Wine```
        
   - Note:
        ```--permutation```    : selects a specific permutation for running multiple
                               experiements with the same permutation when using a feature split
                               example: ```--permutation``` [0,6,4,5,7,2,1,3] for a data set with 8 features when
                               using ```--featuresplit 6``` always uses 0,6,4,5,7,2 in training
        
        ```--checkpoint "./Run"``` : directory for saving checkpoint and model configuration
   Data set names:
        Wine, Telescope, Abalone, Heart
        
        *Heart includes "Heart Disease" and "Diabetes"
        
   WANRNING!
            This code is supported only for Ubuntu 16.04 and 18.04. To run it in another OS it might need some
            changes in the original code.
