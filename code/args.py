"""
Command-line argument parsing.
"""

import argparse
from functools import partial

import time
import tensorflow as tf
import json

from .reptile import Reptile

datasets = [
    "letter",
    "balance-scale",
    "mfeat-morphological",
    "cmc",
    "pendigits",
    "diabetes",
    "tic-tac-toe",
    "vehicle",
    "electricity",
    "wine_quality",
    "vowel",
    "analcatdata_dmft",
    "kc2",
    "kc1",
    "pc1",
    "MagicTelescope",
    "banknote-authentication",
    "blood-transfusion-service-center",
    "ilpd",
    "phoneme",
    "wall-robot-navigation",
    "wdbc",
    "abalone",
    "abalone2",
    "GesturePhaseSegmentationProcessed",
    "numerai28.6",
    "car",
    "steel-plates-fault",
    "wilt",
    "segment",
    "climate-model-simulation-crashes",
    "jungle_chess_2pcs_raw_endgame_complete",
    "codrna",
    "mnist",
    "hearts"
]

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    file_time = int(time.time())
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed',           help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint',     help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--save_path',      help='checkpoint directory', default='exp_'+str(file_time))
    parser.add_argument('--num_jobs',       help='Number of jobs to run in parallel', default=5, type=int)
    #parser.add_argument('--classes',        help='number of classes per inner task', default=5, type=int)

    parser.add_argument('--inner_batch',    help='inner batch size', default=30, type=int)
    parser.add_argument('--inner_iters',    help='inner iterations', default=10, type=int)
    parser.add_argument('--learning_rate',  help='Adam step size', default=0.0001, type=float)
    parser.add_argument('--meta_step',      help='meta-training step size', default=0.01, type=float)
    parser.add_argument('--meta_batch',     help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta_iters',     help='meta-training iterations', default=15001, type=int)
    parser.add_argument('--min_feats',      help='Min number of features', default="4", type=str)
    parser.add_argument('--max_feats',      help='Max number of features', default="8", type=str)


    parser.add_argument('--freeze',         help='whether a permuting network is added', type=boolean_string, default=False)
    parser.add_argument('--conv_layers',    help='Number and size of conv layers',type=str,  default="[8,16,14]")
    parser.add_argument('--base_layers',    help='Number and size of base layers',type=str,  default="[64,64]")
    
    parser.add_argument('--perm_epochs',    help='training epochs for permuter', default=501, type=int)
    parser.add_argument('--perm_lr',        help='permuter learning rate', default=0.0001, type=float)
 
    parser.add_argument('--num_test_features',  help='Ratio of feature split for train test',type=str,  default="0")
    parser.add_argument('--test_feat_ratio',  help='Ratio of feature split for train test',type=float,  default=0.)
    parser.add_argument('--name',           help='name add-on',      type=str,  default='Model_config-'+str(file_time))

    parser.add_argument('--dataset',        help='data set to evaluate on',      type=str,  default='diabetes')
    parser.add_argument('--data_dir',       help='Path to datasets',      type=str,  default='./Data/selected')
    parser.add_argument('--config',         help='json config file', type=str,  default=None)

    args = vars(parser.parse_args())
    if args['dataset'] not in datasets:
        raise ValueError("Dataset argument must be one of openml cc18")

    if args['config'] is None:
        args['config'] = f"{args['checkpoint']}/{args['name']}.json"
        print(args['config'])
        with open(args['config'], 'w') as write_file:
            json.dump(args, write_file)
    else:
        with open(args['config'], 'r') as open_file:
            args = json.load(open_file)
    return parser

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        #'num_classes':          parsed_args.classes,
        'inner_batch_size':     parsed_args.inner_batch,
        'inner_iters':          parsed_args.inner_iters,
        'learning_rate':        parsed_args.learning_rate,
        'meta_step_size':       parsed_args.meta_step,
        'min_feats':            parsed_args.min_feats,
        'max_feats':            parsed_args.max_feats,
        #'feats':                parsed_args.feats,

        'meta_batch_size':      parsed_args.meta_batch,
        'meta_iters':           parsed_args.meta_iters,

        'perm_lr':              parsed_args.perm_lr,
        'perm_epochs':          parsed_args.perm_epochs,
        'name':					parsed_args.name,
        'reptile_fn':           _args_reptile(parsed_args),
        "save_path":            parsed_args.save_path

    }

def _args_reptile(parsed_args):
    return Reptile
