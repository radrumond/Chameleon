"""
Command-line argument parsing.
"""

import argparse
from functools import partial

import time
import tensorflow as tf
import json

from .reptile import Reptile

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
    parser.add_argument('--classes',        help='number of classes per inner task', default=1, type=int)
    parser.add_argument('--shots',          help='number of examples per class', default=60, type=int)
    parser.add_argument('--inner_batch',    help='inner batch size', default=30, type=int)
    parser.add_argument('--inner_iters',    help='inner iterations', default=10, type=int)
    parser.add_argument('--learning_rate',  help='Adam step size', default=0.0001, type=float)
    parser.add_argument('--meta_step',      help='meta-training step size', default=0.01, type=float)
    parser.add_argument('--meta_step_final',help='meta-training step size by the end',default=0, type=float)
    parser.add_argument('--meta_batch',     help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta_iters',     help='meta-training iterations', default=15001, type=int)
    parser.add_argument('--min_class',      help='Min number of features', default=4, type=int)
    parser.add_argument('--max_class',      help='Max number of features', default=8, type=int)
    parser.add_argument('--reg_columns',    help='regularize softmax columns to 1', default=0.0, type=float)
    parser.add_argument('--weight_decay',   help='weight decay rate', default=1, type=float)
    parser.add_argument('--permuter',       help='whether a permuting network is added', type=boolean_string, default=False)
    parser.add_argument('--num_filter',     help='number of filter in the permuter conv layer',default=32, type=int)
    parser.add_argument('--num_conv',       help='number of conv layers in the permuter',default=1, type=int)
    parser.add_argument('--num_neurons',    help='number of neurons in the final dense layer',default=64, type=int)
    parser.add_argument('--conv_act',       help='whether relu activations are used after permuter conv layers', type=boolean_string, default=True)
    parser.add_argument('--perm_epochs',    help='training epochs for permuter', default=501, type=int)
    parser.add_argument('--perm_lr',        help='permuter learning rate', default=0.00001, type=float)
    parser.add_argument('--bootstrap',      help='whether to sample in bootstrapping fashion if feature split is used', type=boolean_string, default=True)
    parser.add_argument('--perm_matrix',    help='whether the output is computed with the permutation matrix or the softmax matrix', type=boolean_string, default=False)
    parser.add_argument('--permutation',    help='Permutation order to split features',type=str,  default="[]")
    parser.add_argument('--feature_split',  help='Ratio of feature split for train test',type=int,  default=0)
    parser.add_argument('--name',           help='name add-on',      type=str,  default='Model_config-'+str(file_time))
    parser.add_argument('--dataset',        help='data set to evaluate on',      type=str,  default='Wine')
    parser.add_argument('--config',         help='json config file', type=str,  default=None)

    args = vars(parser.parse_args())
    if args['dataset'] not in ["Wine","Abalone","Telescope","Heart"]:
        raise ValueError("Dataset argument must be one of: Wine, Abalone, Telescope, Heart")

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
        'num_classes':          parsed_args.classes,
        'num_shots':            parsed_args.shots,
        'inner_batch_size':     parsed_args.inner_batch,
        'inner_iters':          parsed_args.inner_iters,
        'learning_rate':        parsed_args.learning_rate,
        'meta_step_size':       parsed_args.meta_step,
        'min_class':            parsed_args.min_class,
        'max_class':            parsed_args.max_class,
        #'feats':                parsed_args.feats,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size':      parsed_args.meta_batch,
        'meta_iters':           parsed_args.meta_iters,
        'weight_decay_rate':    parsed_args.weight_decay,
        'perm_lr':              parsed_args.perm_lr,
        'perm_epochs':          parsed_args.perm_epochs,
        'feature_split':        parsed_args.feature_split,
        'permutation':			parsed_args.permutation,
        'bootstrap':            parsed_args.bootstrap,
        'name':					parsed_args.name,
        'dataset':				parsed_args.dataset,
        'reptile_fn':           _args_reptile(parsed_args)

    }

def _args_reptile(parsed_args):
    return Reptile
