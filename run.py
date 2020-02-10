## Code modified by: Lukas Brinkmeyer and Rafael Rego Drumond
## Original code at: https://github.com/openai/supervised-reptile

import random
import tensorflow as tf
import os
import ast

from code.args   import argument_parser, train_kwargs
from code.models import BaseModel, Chameleon
from code.train import train

from openmldataset.generator import genFewShot

"""
    Main.
    Initialize the models and run the Chameleon Experiment
"""

def main():
    
    args = argument_parser().parse_args()
    random.seed(args.seed)

    print("########## argument sheet ########################################")
    for arg in vars(args):
        print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
    print("##################################################################")

    # Select data set
    dataset = genFewShot(os.path.join(args.data_dir,args.dataset))
    testFeats = list(dataset.setFeatures(ast.literal_eval(args.num_test_features), ratio=args.test_feat_ratio))
    dataset.splitTrainVal(.25)
    # Creating fixed set of test tasks
    dataset.generateDataset(dataset.totalLabels, ast.literal_eval(args.min_feats), ast.literal_eval(args.max_feats), args.inner_batch, args.inner_batch, args.meta_batch, number_of_mbs=100, test=True, oracle=True)
    
    feats = len(dataset.train_f)

    if not dataset.special:
        print("###################################### Reptile Oracle ####################################")
        # Evaluate Reptile only with already realigned tasks
        tf.reset_default_graph()
        mEnc = None
        model_input = None
        oracle = True
        feats = dataset.totalFeatures

        repModel  = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=model_input,name_suf="RepMod")
        with tf.Session() as sess:
            print('Training...')

            train(sess, repModel, None, mEnc, args.checkpoint,dataset, oracle,False, **train_kwargs(args), name_affix="Reptile_Oracle")


    #### Train and evaluate Chameleon + Reptile
    print("############################### Chameleon + Reptile #####################################")
    # Create Chameleon component
    tf.reset_default_graph()
    print("Initializing Chameleon")
    mEnc  = Chameleon(num_instances=args.inner_batch*dataset.totalLabels,maxK=feats,activation=tf.nn.relu,name_suf="Perm",conv_layers=ast.literal_eval(args.conv_layers))
    mEnc.build()
    model_input = mEnc.out
    oracle = False



    # Initialize base model ŷ for reptile and scratch training
    repModel  = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=model_input,name_suf="RepMod")
    evalModel = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=None       ,name_suf="EvalMod")

    with tf.Session() as sess:
        print('Training...')
        try:
            train(sess, repModel,evalModel,mEnc,args.checkpoint,dataset, oracle,False,**train_kwargs(args), name_affix="Reptile_Chameleon")
        except TypeError:
            raise

    print("####################### Untrained Chameleon + Reptile #####################################")
    tf.reset_default_graph()
    # Create Chameleon component
    print("Initializing Chameleon")
    mEnc  = Chameleon(num_instances=args.inner_batch*dataset.totalLabels,maxK=feats,activation=tf.nn.relu,name_suf="Perm",conv_layers=ast.literal_eval(args.conv_layers))
    mEnc.build()
    model_input = mEnc.out
    oracle = False

    # Initialize base model ŷ for reptile and scratch training
    repModel  = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=model_input,name_suf="RepMod",untrained=True)
    evalModel = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=None       ,name_suf="EvalMod")

    with tf.Session() as sess:
        print('Training...')
        try:
            param_dic = train_kwargs(args)
            param_dic['perm_epochs'] = 0
            train(sess, repModel,evalModel,mEnc,args.checkpoint,dataset, oracle,False,**param_dic, name_affix="Repltile_Chameleon-Unt")
        except TypeError:
            raise

    if args.freeze:
        print("######################## Frozen Chameleon + Reptile #####################################")
        # Create Chameleon component
        tf.reset_default_graph()
        print("Initializing Chameleon")
        mEnc  = Chameleon(num_instances=args.inner_batch*dataset.totalLabels,maxK=feats,activation=tf.nn.relu,name_suf="Perm",conv_layers=ast.literal_eval(args.conv_layers))
        mEnc.build()
        model_input = mEnc.out
        oracle = False



        # Initialize base model ŷ for reptile and scratch training
        repModel  = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=model_input,name_suf="RepMod",frozen=True)
        evalModel = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=None       ,name_suf="EvalMod")

        with tf.Session() as sess:
            print('Training...')
            try:
                train(sess, repModel,evalModel,mEnc,args.checkpoint,dataset, oracle, args.freeze, **train_kwargs(args), name_affix="Reptile_Chameleon-Froz")
            except TypeError:
                raise

    print("###################################### Reptile Padded ####################################")
    # Evaluate Reptile only with padded tasks
    tf.reset_default_graph()
    mEnc = None
    model_input = None
    repModel  = BaseModel(size=feats,layers=ast.literal_eval(args.base_layers),num_classes=dataset.totalLabels,input_ph=model_input,name_suf="RepMod")
    with tf.Session() as sess:
        print('Training...')
        try:
            train(sess, repModel, None, mEnc, args.checkpoint,dataset, oracle,False,**train_kwargs(args), name_affix="Reptile_Pad")
        except TypeError:
            raise


if __name__ == '__main__':
    main()